//! Per-request router-v2 routing for `serve` (spec §9.4).
//!
//! A file with a router policy (bit ROUTER_V2) decides every request on its
//! LAST user message with the backbone-gated rule. The backbone lane is the
//! server's ordinary slot pool; each routed skill gets its own lane — a
//! [`PipelinePool`] of pipelines loaded WITH that skill
//! (`Pipeline::from_model_with_skill`), created on first use and then kept.
//! Nothing ever calls `set_active_skill`: switching the overlay of a slot
//! would invalidate its sequence state, while separate lanes each keep their
//! own KV / ring / recurrent state and their own `kv_prefix` reuse.
//!
//! φ is computed on a DEDICATED backbone pipeline: the φ prefill resets
//! the sequence state of the pipeline it runs on, and doing that on a
//! generation slot would throw away its conversation prefix.

use crate::PipelinePool;
use cortiq_core::CmfModel;
use cortiq_engine::Pipeline;
use cortiq_engine::lookup::{self, LookupMode, LookupOutcome, LookupTable, LookupTables};
use cortiq_engine::router::{self, RouteDecision, RouteOptions, RouteTarget};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

/// Builds a generation pipeline with a skill overlaid (the server applies
/// its own `--o1` / multi-GPU plan here).
pub type LaneFactory = Box<dyn Fn(&str) -> Result<Pipeline, String> + Send + Sync>;

pub struct SkillRouter {
    model: Arc<CmfModel>,
    probe: Mutex<Pipeline>,
    lanes: Mutex<BTreeMap<String, Arc<PipelinePool>>>,
    factory: LaneFactory,
    opts: RouteOptions,
    /// The file's `lookup` records (spec §9.5.2), opened on first use. A
    /// lookup target never gets a lane: the table answers, or the
    /// backbone slots run (with the card prepended in `context` mode).
    lookups: LookupTables,
    lookup_mode: LookupMode,
}

impl SkillRouter {
    /// `probe` must be a plain backbone pipeline of `model` that serves
    /// nothing else.
    pub fn new(
        model: Arc<CmfModel>,
        probe: Pipeline,
        factory: LaneFactory,
        include_quarantine: bool,
    ) -> Self {
        Self {
            lookups: LookupTables::new(model.clone()),
            model,
            probe: Mutex::new(probe),
            lanes: Mutex::new(BTreeMap::new()),
            factory,
            opts: RouteOptions { include_quarantine },
            lookup_mode: LookupMode::default(),
        }
    }

    /// What a request routed to a lookup record does (`--lookup-mode`,
    /// `CMF_LOOKUP_MODE`; default `answer`).
    pub fn with_lookup_mode(mut self, mode: LookupMode) -> Self {
        self.lookup_mode = mode;
        self
    }

    pub fn lookup_mode(&self) -> LookupMode {
        self.lookup_mode
    }

    /// Does the file carry at least one lookup record? Then every routed
    /// request takes the lookup pre-pass ([`Self::decide_lookup`]).
    pub fn has_lookups(&self) -> bool {
        self.lookups.any()
    }

    /// Ids of the file's lookup records.
    pub fn lookup_ids(&self) -> Vec<String> {
        self.lookups.ids()
    }

    /// The table of lookup record `id` (opened on first use); `None` when
    /// `id` is not a lookup record.
    pub fn lookup_table(&self, id: &str) -> Result<Option<Arc<LookupTable>>, String> {
        self.lookups.get(id)
    }

    /// [`Self::decide_framed`], then the lookup step: a decision for a
    /// lookup record becomes its outcome (`lookup::resolve_lookup`) — no
    /// key in the message sends the request to the backbone unchanged.
    pub fn decide_lookup(
        &self,
        user_text: &str,
        frame: router::PromptFrame,
    ) -> Result<(RouteDecision, LookupOutcome), String> {
        self.decide_lookup_turns(&[user_text], frame)
    }

    /// [`Self::decide_lookup`] with conversation memory: `turns` are the
    /// user messages, the LAST one first — the routing decision (φ) is
    /// made on it alone — then the earlier ones back in time (the chat
    /// endpoint passes up to `lookup::MEMORY_TURNS`); the key comes from
    /// the most recent turn that holds one, the field and the language
    /// from the last (`lookup::resolve_lookup_gated`). The decision is the
    /// router's, so a `key_first` record may take a backbone decision on
    /// a STRONG key of the last message (never of an earlier turn) —
    /// `decided_by: "key_first"` in `x_cortiq_route`. Empty `turns` = the
    /// backbone, nothing looked up.
    pub fn decide_lookup_turns(
        &self,
        turns: &[&str],
        frame: router::PromptFrame,
    ) -> Result<(RouteDecision, LookupOutcome), String> {
        let last = turns.first().copied().unwrap_or("");
        let d = self.decide_framed(last, frame);
        let gate = lookup::KeyFirstGate::new(self.opts, frame);
        lookup::resolve_lookup_gated(&self.lookups, d, Some(gate), turns, self.lookup_mode)
    }

    pub fn model(&self) -> &Arc<CmfModel> {
        &self.model
    }

    /// The decision for one user message (blocking: a φ prefill up to
    /// `router.phi.layer`). Concurrent requests serialize on the probe.
    pub fn decide(&self, user_text: &str) -> RouteDecision {
        let mut probe = self.probe.lock().unwrap_or_else(|e| e.into_inner());
        router::route_request_with(&self.model, &mut probe, user_text, self.opts)
    }

    /// The lane of a skill target (blocking on first use: loads one
    /// pipeline with the skill overlaid). `None` for the backbone — the
    /// caller's own slot pool.
    pub fn lane(&self, target: &RouteTarget) -> Result<Option<Arc<PipelinePool>>, String> {
        let RouteTarget::Skill(id) = target else {
            return Ok(None);
        };
        // A lookup record has no lane: the backbone slots serve it (the
        // table never touches the network).
        if LookupTable::is_lookup(&self.model, id) {
            return Ok(None);
        }
        let mut lanes = self.lanes.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(pool) = lanes.get(id) {
            return Ok(Some(pool.clone()));
        }
        let pipeline = (self.factory)(id).map_err(|e| format!("skill lane '{id}': {e}"))?;
        tracing::info!("route: skill lane '{id}' loaded (first request routed to it)");
        let pool = Arc::new(PipelinePool::new(vec![pipeline]));
        lanes.insert(id.clone(), pool.clone());
        Ok(Some(pool))
    }

    /// Skill lanes loaded so far (sorted ids).
    pub fn loaded_lanes(&self) -> Vec<String> {
        self.lanes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .keys()
            .cloned()
            .collect()
    }

    /// Skills the decision can pick (auto-routable, or any non-retired
    /// class with the debug flag) — the engine's own rule.
    pub fn routable_skills(&self) -> Vec<String> {
        router::routable_skills(&self.model.header, self.opts)
            .into_iter()
            .map(|s| s.id.clone())
            .collect()
    }

    /// The decision for one request with its prompt frame: [`Self::decide`]
    /// on `user_text`, then the chosen skill's prompt contract against the
    /// `frame` the lane will generate with (`serve` never re-renders: a
    /// skill whose contract the frame does not satisfy runs the backbone).
    pub fn decide_framed(&self, user_text: &str, frame: router::PromptFrame) -> RouteDecision {
        let d = self.decide(user_text);
        router::enforce_prompt_contract(&self.model.header, d, frame, false).0
    }
}

/// What a `/v1/completions` prompt routes on: the user text of a prompt
/// that is EXACTLY one cmf-im-v1 user turn + generation prompt (its
/// frame then satisfies a cmf-im-v1 skill's contract and φ is the
/// canonical one); anything else — raw text, a pre-rendered history, a
/// system turn — has no single user turn to route on and runs the
/// backbone (R5/PHI-2: never a φ over a whole transcript under one probe
/// lock, never a skill on a prompt outside its contract).
pub fn completions_route_text(prompt: &str) -> Option<String> {
    router::cmf_im_v1_single_user_turn(prompt).map(str::to_string)
}

/// Reason reported when a `/v1/completions` prompt is not routed.
pub const COMPLETIONS_NO_USER_TURN: &str =
    "completions: no user turn (the prompt is not exactly one cmf-im-v1 user turn) — the \
     backbone runs";

/// The text a chat request is routed on: the LAST `user` message.
pub fn last_user_text<'a>(
    messages: impl DoubleEndedIterator<Item = (&'a str, String)>,
) -> Option<String> {
    messages
        .rev()
        .find(|(role, _)| *role == "user")
        .map(|(_, text)| text)
}

/// The last `max` `user` messages of a chat, the LAST one first (the
/// lookup pre-pass routes on `[0]` and remembers the key of an earlier
/// turn: `lookup::MEMORY_TURNS`).
pub fn recent_user_texts<'a>(
    messages: impl DoubleEndedIterator<Item = (&'a str, String)>,
    max: usize,
) -> Vec<String> {
    messages
        .rev()
        .filter(|(role, _)| *role == "user")
        .map(|(_, text)| text)
        .take(max)
        .collect()
}
