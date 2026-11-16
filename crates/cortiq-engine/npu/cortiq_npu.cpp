// cortiq-npu: a thin C ABI over Qualcomm's QNN HTP backend (Hexagon NPU).
//
// One matrix = one graph: a 1x1 Conv2d with int8 weights quantized per output
// row and 16-bit activations. The caller owns the fixed-point conversion of
// the activation; this file only moves tensors. Built separately from the
// engine (needs the QAIRT SDK headers), loaded with dlopen at run time.
#include <dlfcn.h>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>
#include "QnnInterface.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpPerfInfrastructure.h"

namespace {
struct Mat { Qnn_ContextHandle_t ctx; Qnn_GraphHandle_t graph; Qnn_Tensor_t in, out; uint32_t rows, cols; uint32_t din[4], dout[4]; };
const QNN_INTERFACE_VER_TYPE *q = nullptr;
Qnn_LogHandle_t g_log = nullptr; Qnn_BackendHandle_t g_be = nullptr; Qnn_DeviceHandle_t g_dev = nullptr;
std::vector<Mat *> g_mats; std::mutex g_mu; char g_err[256] = "";
void logcb(const char *fmt, QnnLog_Level_t lvl, uint64_t, va_list a) { if (lvl <= QNN_LOG_LEVEL_ERROR) { vfprintf(stderr, fmt, a); fputc('\n', stderr); } }
int fail(const char *what, Qnn_ErrorHandle_t e) { snprintf(g_err, sizeof g_err, "%s -> %llu", what, (unsigned long long)e); return -1; }
Qnn_Tensor_t tensor(const char *name, Qnn_TensorType_t type, Qnn_DataType_t dt, uint32_t *dims, uint32_t rank, void *data, uint32_t size) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT; t.version = QNN_TENSOR_VERSION_1; t.v1.name = name; t.v1.type = type; t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  t.v1.dataType = dt; t.v1.rank = rank; t.v1.dimensions = dims; t.v1.memType = QNN_TENSORMEMTYPE_RAW; t.v1.clientBuf.data = data; t.v1.clientBuf.dataSize = size; return t; }
void so(Qnn_Tensor_t &t, float scale, int32_t offset) { t.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED; t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  t.v1.quantizeParams.scaleOffsetEncoding.scale = scale; t.v1.quantizeParams.scaleOffsetEncoding.offset = offset; }
}  // namespace

extern "C" {
const char *cq_npu_error() { return g_err; }

// Bring the backend up in burst power mode. `lib` is libQnnHtp.so (or a path).
int cq_npu_init(const char *lib) {
  std::lock_guard<std::mutex> l(g_mu);
  if (q) return 0;
  void *h = dlopen(lib ? lib : "libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
  if (!h) { snprintf(g_err, sizeof g_err, "dlopen: %s", dlerror()); return -1; }
  auto getp = (Qnn_ErrorHandle_t (*)(const QnnInterface_t ***, uint32_t *))dlsym(h, "QnnInterface_getProviders");
  const QnnInterface_t **prov = nullptr; uint32_t n = 0;
  if (!getp || getp(&prov, &n) != QNN_SUCCESS || !n) { snprintf(g_err, sizeof g_err, "no QNN providers"); return -1; }
  const auto *iq = &prov[0]->QNN_INTERFACE_VER_NAME;
  iq->logCreate(logcb, QNN_LOG_LEVEL_ERROR, &g_log);
  Qnn_ErrorHandle_t e = iq->backendCreate(g_log, nullptr, &g_be); if (e != QNN_SUCCESS) return fail("backendCreate", e);
  if (iq->deviceCreate && iq->deviceCreate(g_log, nullptr, &g_dev) != QNN_SUCCESS) g_dev = nullptr;
  QnnDevice_Infrastructure_t infra = nullptr;
  if (iq->deviceGetInfrastructure && iq->deviceGetInfrastructure(&infra) == QNN_SUCCESS && infra) {
    auto perf = ((QnnHtpDevice_Infrastructure_t *)infra)->perfInfra; uint32_t id = 0;
    if (perf.createPowerConfigId && perf.createPowerConfigId(0, 0, &id) == QNN_SUCCESS) {
      QnnHtpPerfInfrastructure_PowerConfig_t c = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
      c.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3; auto &d = c.dcvsV3Config;
      d.contextId = id; d.setDcvsEnable = 1; d.dcvsEnable = 0; d.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
      d.setSleepLatency = 1; d.sleepLatency = 40; d.setBusParams = 1; d.setCoreParams = 1;
      d.busVoltageCornerMin = d.busVoltageCornerTarget = d.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      d.coreVoltageCornerMin = d.coreVoltageCornerTarget = d.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      const QnnHtpPerfInfrastructure_PowerConfig_t *arr[] = {&c, nullptr}; perf.setPowerConfig(id, arr);
    }
  }
  q = iq; return 0;
}

// Register a rows x cols matrix. `w` is int8 row-major; row r's true weights
// are w[r][c] * row_scale[r]. Activation in: u16, real = (v - 32768) / 32768.
// Out: u16, real = (v - 32768) * out_scale. Returns a handle >= 0.
int cq_npu_add(uint32_t rows, uint32_t cols, const int8_t *w, const float *row_scale, float out_scale) {
  if (!q) { snprintf(g_err, sizeof g_err, "not initialised"); return -1; }
  std::lock_guard<std::mutex> l(g_mu);
  auto *m = new Mat(); m->rows = rows; m->cols = cols; Qnn_ErrorHandle_t e;
  if ((e = q->contextCreate(g_be, g_dev, nullptr, &m->ctx)) != QNN_SUCCESS) return fail("contextCreate", e);
  if ((e = q->graphCreate(m->ctx, "mv", nullptr, &m->graph)) != QNN_SUCCESS) return fail("graphCreate", e);
  // HWIO: [1, 1, cols, rows] — transpose the row-major matrix.
  std::vector<int8_t> hwio((size_t)rows * cols);
  for (uint32_t r = 0; r < rows; r++) for (uint32_t c = 0; c < cols; c++) hwio[(size_t)c * rows + r] = w[(size_t)r * cols + c];
  std::vector<Qnn_ScaleOffset_t> sos(rows); for (uint32_t r = 0; r < rows; r++) { sos[r].scale = row_scale[r] > 0 ? row_scale[r] : 1e-12f; sos[r].offset = 0; }
  m->din[0] = m->din[1] = m->din[2] = 1; m->din[3] = cols; m->dout[0] = m->dout[1] = m->dout[2] = 1; m->dout[3] = rows;
  uint32_t dw[4] = {1, 1, cols, rows}, d2[1] = {2}, d22[2] = {2, 2}; uint32_t stride[2] = {1, 1}, pad[4] = {0, 0, 0, 0};
  m->in = tensor("x", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_UFIXED_POINT_16, m->din, 4, nullptr, 0); so(m->in, 1.0f / 32768.0f, -32768);
  m->out = tensor("y", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_UFIXED_POINT_16, m->dout, 4, nullptr, 0); so(m->out, out_scale, -32768);
  Qnn_Tensor_t tw = tensor("w", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_SFIXED_POINT_8, dw, 4, hwio.data(), (uint32_t)hwio.size());
  tw.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED; tw.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
  tw.v1.quantizeParams.axisScaleOffsetEncoding.axis = 3; tw.v1.quantizeParams.axisScaleOffsetEncoding.numScaleOffsets = rows; tw.v1.quantizeParams.axisScaleOffsetEncoding.scaleOffset = sos.data();
  Qnn_Tensor_t ts = tensor("stride", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_UINT_32, d2, 1, stride, sizeof stride);
  Qnn_Tensor_t tp = tensor("pad", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_UINT_32, d22, 2, pad, sizeof pad);
  if ((e = q->tensorCreateGraphTensor(m->graph, &m->in)) != QNN_SUCCESS) return fail("tensor x", e);
  if ((e = q->tensorCreateGraphTensor(m->graph, &tw)) != QNN_SUCCESS) return fail("tensor w", e);
  if ((e = q->tensorCreateGraphTensor(m->graph, &ts)) != QNN_SUCCESS) return fail("tensor stride", e);
  if ((e = q->tensorCreateGraphTensor(m->graph, &tp)) != QNN_SUCCESS) return fail("tensor pad", e);
  if ((e = q->tensorCreateGraphTensor(m->graph, &m->out)) != QNN_SUCCESS) return fail("tensor y", e);
  Qnn_Param_t params[2] = {QNN_PARAM_INIT, QNN_PARAM_INIT};
  params[0].paramType = QNN_PARAMTYPE_TENSOR; params[0].name = "stride"; params[0].tensorParam = ts;
  params[1].paramType = QNN_PARAMTYPE_TENSOR; params[1].name = "pad_amount"; params[1].tensorParam = tp;
  Qnn_Tensor_t ins[2] = {m->in, tw}; Qnn_Tensor_t outs[1] = {m->out};
  Qnn_OpConfig_t op = QNN_OPCONFIG_INIT; op.version = QNN_OPCONFIG_VERSION_1; op.v1.name = "conv"; op.v1.packageName = "qti.aisw"; op.v1.typeName = "Conv2d";
  op.v1.numOfParams = 2; op.v1.params = params; op.v1.numOfInputs = 2; op.v1.inputTensors = ins; op.v1.numOfOutputs = 1; op.v1.outputTensors = outs;
  if ((e = q->graphAddNode(m->graph, op)) != QNN_SUCCESS) return fail("graphAddNode", e);
  if ((e = q->graphFinalize(m->graph, nullptr, nullptr)) != QNN_SUCCESS) return fail("graphFinalize", e);
  g_mats.push_back(m); return (int)g_mats.size() - 1;
}

// y = W x in the fixed-point forms `cq_npu_add` describes. Blocking.
int cq_npu_run(int handle, const uint16_t *x, uint16_t *y) {
  if (handle < 0 || (size_t)handle >= g_mats.size()) { snprintf(g_err, sizeof g_err, "bad handle"); return -1; }
  Mat *m = g_mats[handle]; Qnn_Tensor_t in = m->in, out = m->out;
  in.v1.clientBuf.data = (void *)x; in.v1.clientBuf.dataSize = m->cols * 2; out.v1.clientBuf.data = y; out.v1.clientBuf.dataSize = m->rows * 2;
  Qnn_ErrorHandle_t e = q->graphExecute(m->graph, &in, 1, &out, 1, nullptr, nullptr);
  return e == QNN_SUCCESS ? 0 : fail("graphExecute", e);
}
}
