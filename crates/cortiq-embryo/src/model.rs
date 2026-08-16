//! Embryo genome: configuration and parameter layout (filled in as the
//! trainer grows; see docs/NATIVE_MODEL_TECH.ru.md §3).

/// Embryo-0 as born (§3.2). All matrix dims are multiples of 64 so the
/// GEMM tile contract holds without edge paths.
#[derive(Clone, Debug)]
pub struct EmbryoCfg {
    pub vocab: usize,
    pub hidden: usize,
    pub layers: usize,
    /// every `anchor_every`-th layer is a softmax anchor (o1-ready)
    pub anchor_every: usize,
    // hybrid_k mixer
    pub heads: usize,
    pub nphase: usize,
    pub dv: usize,
    // anchor (GQA softmax)
    pub anchor_q_heads: usize,
    pub anchor_kv_heads: usize,
    pub anchor_hd: usize,
    // experts
    pub experts: usize,
    pub inter: usize,
    // hierarchical head
    pub head_clusters: usize,
    pub mtp_heads: usize,
    pub seq: usize,
}

impl EmbryoCfg {
    pub fn embryo0() -> Self {
        EmbryoCfg {
            vocab: 32768,
            hidden: 384,
            layers: 8,
            anchor_every: 8,
            heads: 8,
            nphase: 32,
            dv: 128,
            anchor_q_heads: 8,
            anchor_kv_heads: 2,
            anchor_hd: 128,
            experts: 4,
            inter: 768,
            head_clusters: 128,
            mtp_heads: 2,
            seq: 1024,
        }
    }
    pub fn is_anchor(&self, layer: usize) -> bool {
        (layer + 1) % self.anchor_every == 0
    }
    /// Parameter count (total, active per token).
    pub fn params(&self) -> (usize, usize) {
        let h = self.hidden;
        let embed = self.vocab * h; // tied lm_head
        let mixer = 2 * (self.heads * self.nphase * h) // thq, thk
            + self.heads * self.dv * h                  // v_proj
            + h * self.heads * self.dv                  // out_proj
            + self.heads * h + self.heads;              // κ gate
        let anchor = self.anchor_q_heads * self.anchor_hd * h
            + 2 * self.anchor_kv_heads * self.anchor_hd * h
            + h * self.anchor_q_heads * self.anchor_hd;
        let ffn_one = 3 * h * self.inter;
        let ffn_total = ffn_one * (self.experts + 1);
        let ffn_active = ffn_one * 2; // top-1 + shared
        let norms = 2 * h;
        let mut total = embed + h;
        let mut active = embed + h;
        for l in 0..self.layers {
            let mix = if self.is_anchor(l) { anchor } else { mixer };
            total += mix + ffn_total + norms;
            active += mix + ffn_active + norms;
        }
        (total, active)
    }
}
