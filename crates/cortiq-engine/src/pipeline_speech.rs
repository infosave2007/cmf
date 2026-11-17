//! Step-by-step access to the language model for LFM3-Audio's speech
//! generation (`crate::lfm3_speech`): a speech frame enters the model as
//! an embedding row (the sum of its codebook embeddings), never as a token,
//! and the depth decoder reads the final-normed hidden state of every step.
//! A child module of `pipeline` so it drives the same layer walk the
//! generation loop runs.

use super::{Pipeline, PrefillIn};
use crate::inference;

impl Pipeline {
    /// Prefill a fresh sequence from `ids` (with their embedding `rows`
    /// when media replace some of them) and return the final-normed hidden
    /// state of the last position.
    pub fn speech_prefill(&mut self, ids: &[u32], rows: Option<&[f32]>) -> Result<Vec<f32>, String> {
        #[cfg(target_os = "macos")]
        crate::gpu_metal::set_io_namespace(self.graph_kv_id);
        if ids.is_empty() {
            return Err("speech: empty prompt".into());
        }
        let hs = self.hidden_size;
        self.clear_sequence_state();
        self.graph_want_logits = false;
        let input = match rows {
            Some(r) if r.len() == ids.len() * hs => PrefillIn::Hidden(r),
            Some(_) => return Err("speech: prompt rows and ids disagree".into()),
            None => PrefillIn::Ids(ids),
        };
        let hidden = self.prefill_input_rows(input, 0, None)?;
        let last = hidden
            .get(hidden.len().saturating_sub(hs)..)
            .filter(|l| l.len() == hs)
            .ok_or("speech: the prefill returned no rows")?;
        Ok(self.final_normed(last))
    }

    /// One position: `row` (an embedding row) at `position`, the cache
    /// holding `[0, position)`; the final-normed hidden state out.
    pub fn speech_step(&mut self, row: &[f32], position: usize) -> Vec<f32> {
        #[cfg(target_os = "macos")]
        crate::gpu_metal::set_io_namespace(self.graph_kv_id);
        self.graph_want_logits = false;
        self.graph_logits = None;
        let hidden = self.forward_layers(row, position, None);
        self.final_normed(&hidden)
    }

    /// One position like [`Self::speech_step`], and the vocabulary logits
    /// of it (a Haru frame needs both). The token graph that folds the head
    /// in reads back the logits *instead of* the hidden state, so the
    /// hidden comes from the plain step and the head runs on its own.
    pub fn speech_step_logits(&mut self, row: &[f32], position: usize) -> (Vec<f32>, Vec<f32>) {
        let normed = self.speech_step(row, position);
        let logits = self.lm_head_forward(&normed);
        (normed, logits)
    }

    /// Start a fresh short sequence on the same buffers: the host caches
    /// emptied, every device K/V cursor back to row 0 (the rows stay
    /// allocated and the next steps overwrite them). The cheap reset of a
    /// stack that runs many tiny sequences — Qwen3-TTS's code predictor,
    /// sixteen positions a frame — where a full reset would free and
    /// rebuild the device mirrors every frame. Exact-attention stacks only:
    /// a recurrent state is not rewound this way.
    pub fn speech_rewind(&mut self) -> bool {
        if self.gdn_cfg.is_some() {
            return false;
        }
        self.kv_cache.clear();
        self.graph_logits = None;
        for li in 0..self.num_layers {
            crate::gpu::graph_kv_set_stored(self.graph_kv_id, li, 0);
        }
        true
    }

    /// The vocabulary logits of a final-normed hidden state.
    pub fn speech_logits(&self, normed: &[f32]) -> Vec<f32> {
        self.lm_head_forward(normed)
    }

    fn final_normed(&self, hidden: &[f32]) -> Vec<f32> {
        let mut out = vec![0f32; self.hidden_size];
        inference::rms_norm_into(hidden, &self.weights.final_norm, self.rms_eps, self.norm_style, &mut out);
        out
    }
}
