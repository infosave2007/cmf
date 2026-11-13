# Compare Edge0 skeleton GGUF tensors with Qwen3.8-Flash-Next bf16 originals.
import json, struct, numpy as np, sys, glob
from gguf import GGUFReader
def st_open(path):
    f=open(path,'rb'); n=struct.unpack('<Q',f.read(8))[0]; h=json.loads(f.read(n)); base=8+n
    return path,h,base
def st_get(sts,name):
    for path,h,base in sts:
        if name in h:
            m=h[name]; a,b=m['data_offsets']; dt=m['dtype']
            raw=np.fromfile(path,dtype=np.uint8,count=b-a,offset=base+a)
            if dt=='BF16': v=(raw.view(np.uint16).astype(np.uint32)<<16).view(np.float32)
            elif dt=='F32': v=raw.view(np.float32)
            elif dt=='F16': v=raw.view(np.float16).astype(np.float32)
            else: raise SystemExit(dt)
            return v.reshape(m['shape'])
    return None
sts=[st_open(p) for p in sorted(glob.glob('/root/e0/src/qwen/*.safetensors'))]
r=GGUFReader('/root/e0/src/e145n/model-skeleton.gguf')
T={t.name:t for t in r.tensors}
def gg(name):
    t=T[name]; ty=t.tensor_type.name; raw=np.asarray(t.data)
    if ty=='Q8_0':
        b=raw.reshape(-1,34); d=b[:,:2].copy().view(np.float16).astype(np.float32).reshape(-1); q=b[:,2:].view(np.int8).astype(np.float32)
        return (q*d[:,None]).reshape(-1), np.repeat(d,32), ty
    if ty=='BF16': return (raw.view(np.uint16).astype(np.uint32)<<16).view(np.float32).reshape(-1), None, ty
    if ty in('F32',): return raw.view(np.float32).reshape(-1), None, ty
    if ty=='F16': return raw.view(np.float16).astype(np.float32).reshape(-1), None, ty
    raise SystemExit(ty)
P='model.language_model.layers.'
pairs=[('blk.0.ffn_gate_inp.weight',P+'0.mlp.gate.weight'),('blk.3.ffn_gate_inp.weight',P+'3.mlp.gate.weight'),
 ('blk.0.ffn_gate_shexp.weight',P+'0.mlp.shared_expert.gate_proj.weight'),('blk.0.ffn_down_shexp.weight',P+'0.mlp.shared_expert.down_proj.weight'),
 ('blk.3.ffn_up_shexp.weight',P+'3.mlp.shared_expert.up_proj.weight'),('blk.0.ffn_gate_inp_shexp.weight',P+'0.mlp.shared_expert_gate.weight'),
 ('blk.3.attn_q.weight',P+'3.self_attn.q_proj.weight'),('blk.3.attn_output.weight',P+'3.self_attn.o_proj.weight'),('blk.3.attn_k.weight',P+'3.self_attn.k_proj.weight'),
 ('blk.0.attn_qkv.weight',P+'0.linear_attn.in_proj_qkv.weight'),('blk.0.ssm_out.weight',P+'0.linear_attn.out_proj.weight'),
 ('blk.0.hc_attn_down.weight',P+'0.attn_hyper_connection.input_mix_weight_down.weight'),('blk.3.hc_ffn_up.weight',P+'3.mlp_hyper_connection.input_mix_weight_up.weight'),
 ('blk.3.attn_q_norm.weight',P+'3.self_attn.q_norm.weight')]
for g,q in pairs:
    if g not in T: print('no gguf',g); continue
    ref=st_get(sts,q)
    if ref is None: print('no ref',q); continue
    v,d,ty=gg(g); ref=ref.reshape(-1).astype(np.float32)
    if v.size!=ref.size: print('size',g,v.size,ref.size); continue
    diff=v-ref; rel=np.sqrt((diff**2).sum()/((ref**2).sum()+1e-30))
    line='%-30s %-5s relL2 %.5f maxabs %.3g'%(g,ty,rel,np.abs(diff).max())
    if d is not None:
        ratio=np.abs(diff)/np.maximum(d,1e-30); line+='  max|err|/d %.3f  frac>0.6d %.2e'%(ratio.max(),(ratio>0.6).mean())
    else: line+='  exact=%s'%(bool((diff==0).all()))
    print(line)
