//! Runtime state management — model, active task, metrics.

use cortiq_core::CmfModel;
use cortiq_core::mask::{MaskCatalog, MaskDiff, TaskMask};
#[cfg(not(target_os = "macos"))]
use cortiq_core::types::SimdType;
use cortiq_core::types::{ExecutionMode, LayerStats, PerformanceMetrics};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

/// Main runtime managing model state, active task, and metrics.
pub struct CortiqRuntime {
    /// Loaded CMF model
    model: Arc<CmfModel>,
    /// Current state (behind RwLock for concurrent reads)
    state: Arc<RwLock<RuntimeState>>,
    /// Start time
    started_at: Instant,
}

/// Mutable runtime state.
#[derive(Debug)]
struct RuntimeState {
    active_task: String,
    active_mask: Option<TaskMask>,
    execution_mode: ExecutionMode,
    metrics: PerformanceMetrics,
    layer_stats: Vec<LayerStats>,
}

/// Response from a task switch operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwitchResult {
    pub previous_task: String,
    pub new_task: String,
    pub switch_mode: String,
    pub switch_latency_ms: f64,
    pub new_sparsity: f32,
    pub new_active_params: String,
    pub diff: MaskDiff,
}

/// Runtime status snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeStatus {
    pub model_name: String,
    pub model_path: String,
    pub format: String,
    pub quantization: String,
    pub execution_mode: ExecutionMode,
    pub active_task: String,
    pub active_sparsity: f32,
    pub active_params: String,
    pub active_layers: usize,
    pub total_layers: usize,
    pub performance: PerformanceMetrics,
    pub layer_stats: Vec<LayerStats>,
}

impl CortiqRuntime {
    /// Create a new runtime from a loaded CMF model.
    pub fn new(model: Arc<CmfModel>) -> Self {
        let arch = model.arch();
        let n_layers = arch.num_layers;

        // Detect execution mode
        let execution_mode = Self::detect_execution_mode();

        // Default layer stats. FFN neuron counts come from the actual
        // gate_proj shape (the directory is the size authority) so a
        // physically-defragged layer (spec §11) reports its true reduced
        // count, not the nominal arch scalar; fall back to the scalar when
        // no dense gate_proj is present (e.g. MoE router layers).
        let ffn_neurons = |i: usize| -> usize {
            model
                .tensor(&format!("model.layers.{i}.mlp.gate_proj.weight"))
                .and_then(|t| t.shape.first().copied())
                .unwrap_or(arch.intermediate_size)
        };
        let layer_stats: Vec<LayerStats> = (0..n_layers)
            .map(|i| LayerStats {
                layer_idx: i,
                active_neurons: ffn_neurons(i),
                total_neurons: ffn_neurons(i),
                active_heads: arch.num_attention_heads,
                total_heads: arch.num_attention_heads,
                is_alive: true,
                placement: "gpu".to_string(),
                avg_forward_ms: 0.0,
            })
            .collect();

        let state = RuntimeState {
            active_task: model.masks.default_task.clone(),
            active_mask: model.masks.fallback().cloned(),
            execution_mode,
            metrics: PerformanceMetrics::default(),
            layer_stats,
        };

        Self {
            model,
            state: Arc::new(RwLock::new(state)),
            started_at: Instant::now(),
        }
    }

    /// Detect optimal execution mode for current hardware.
    fn detect_execution_mode() -> ExecutionMode {
        // TODO: actual hardware detection (CUDA, Metal, CPU caps)
        #[cfg(target_os = "macos")]
        {
            ExecutionMode::AppleUnified {
                metal_layers: vec![],
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            ExecutionMode::CpuOnly {
                simd_type: SimdType::Avx2,
                threads: num_cpus(),
            }
        }
    }

    /// Switch to a different task mask.
    pub async fn switch_task(&self, task_name: &str) -> Result<SwitchResult, anyhow::Error> {
        let new_mask = self
            .model
            .masks
            .get(task_name)
            .ok_or_else(|| anyhow::anyhow!("Task mask '{}' not found", task_name))?
            .clone();

        let mut state = self.state.write().await;
        let previous_task = state.active_task.clone();

        // Spec §5.1: switching onto an unmeasured mask must be loud.
        if new_mask.quality.is_none() {
            tracing::warn!(
                "task '{task_name}': mask has no measured quality — \
                 treat outputs as unvalidated"
            );
        }

        let switch_start = Instant::now();

        // Compute diff for efficient swap
        let diff = if let Some(ref current) = state.active_mask {
            current.diff(&new_mask)
        } else {
            MaskDiff {
                changed_layers: (0..self.model.arch().num_layers).collect(),
                neurons_added: 0,
                neurons_removed: 0,
                ffn_delta: vec![],
            }
        };

        // Update layer stats based on new mask
        for ls in &mut state.layer_stats {
            ls.active_neurons = new_mask.ffn_active_count(ls.layer_idx);
            ls.active_heads = new_mask.active_head_count(ls.layer_idx);
            ls.is_alive = new_mask.layer_alive(ls.layer_idx);
        }

        let switch_latency = switch_start.elapsed().as_secs_f64() * 1000.0;

        state.active_task = task_name.to_string();
        let sparsity = new_mask.sparsity;
        state.active_mask = Some(new_mask);
        state.metrics.last_switch_latency_ms = switch_latency;
        state.metrics.total_switches += 1;

        let active_params = self.estimate_active_params(&state);

        Ok(SwitchResult {
            previous_task,
            new_task: task_name.to_string(),
            switch_mode: "warm".to_string(),
            switch_latency_ms: switch_latency,
            new_sparsity: sparsity,
            new_active_params: active_params,
            diff,
        })
    }

    /// Get current runtime status.
    pub async fn status(&self) -> RuntimeStatus {
        let state = self.state.read().await;
        let mut metrics = state.metrics.clone();
        metrics.uptime_seconds = self.started_at.elapsed().as_secs();

        let active_sparsity = state
            .active_mask
            .as_ref()
            .map(|m| m.sparsity)
            .unwrap_or(0.0);

        let active_layers = state.layer_stats.iter().filter(|l| l.is_alive).count();

        RuntimeStatus {
            model_name: self.model.arch().arch_name.clone(),
            model_path: self.model.path.display().to_string(),
            format: format!("CMF v{}", self.model.header.version),
            quantization: format!("{:?}", self.model.header.quant_type),
            execution_mode: state.execution_mode.clone(),
            active_task: state.active_task.clone(),
            active_sparsity,
            active_params: self.estimate_active_params(&state),
            active_layers,
            total_layers: self.model.arch().num_layers * self.model.arch().num_loops,
            performance: metrics,
            layer_stats: state.layer_stats.clone(),
        }
    }

    /// Get mask catalog.
    pub fn masks(&self) -> &MaskCatalog {
        &self.model.masks
    }

    /// Get model reference.
    pub fn model(&self) -> &CmfModel {
        &self.model
    }

    /// Snapshot of the currently active task mask (None = dense).
    pub async fn active_mask(&self) -> Option<TaskMask> {
        self.state.read().await.active_mask.clone()
    }

    /// Atomically snapshot the active task name and its mask. Request paths
    /// must use one snapshot rather than reading these fields separately.
    pub async fn active_selection(&self) -> (String, Option<TaskMask>) {
        let state = self.state.read().await;
        (state.active_task.clone(), state.active_mask.clone())
    }

    /// Record a finished generation into the performance metrics.
    pub async fn record_generation(&self, gen_tokens: usize, elapsed_ms: f64, ttft_ms: f64) {
        let mut state = self.state.write().await;
        let m = &mut state.metrics;
        let prev_total = m.tokens_generated as f64;
        m.tokens_generated += gen_tokens as u64;
        if elapsed_ms > 0.0 {
            let tps = gen_tokens as f64 / (elapsed_ms / 1000.0);
            // Running average weighted by token counts.
            let total = prev_total + gen_tokens as f64;
            m.avg_tokens_per_sec = if total > 0.0 {
                (m.avg_tokens_per_sec * prev_total + tps * gen_tokens as f64) / total
            } else {
                tps
            };
        }
        if m.avg_time_to_first_token_ms == 0.0 {
            m.avg_time_to_first_token_ms = ttft_ms;
        } else {
            m.avg_time_to_first_token_ms = m.avg_time_to_first_token_ms * 0.9 + ttft_ms * 0.1;
        }
    }

    /// Estimate active parameters string (from real tensor shapes).
    fn estimate_active_params(&self, state: &RuntimeState) -> String {
        let total_params = self.model.total_param_count() as f64 / 1e9;
        let sparsity = state
            .active_mask
            .as_ref()
            .map(|m| m.sparsity as f64)
            .unwrap_or(0.0);
        let active = total_params * (1.0 - sparsity);
        format!("{:.2}B / {:.2}B", active, total_params)
    }
}

#[cfg(not(target_os = "macos"))]
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}
