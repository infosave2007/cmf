//! GDN mixer, chunked WY/UT form (plan S7, level 2 — Vulkan only): the same
//! operator as the token scan (`gdn_scan_fwd/bwd`), computed per 64-token
//! chunk in GEMM form so the sequential part is one small GEMM chain per
//! chunk instead of one state pass per token.
//!
//! Per chunk (local token i, chunk start state S_0, α_t = per-token decay):
//! ```text
//! lg_i = Σ_{j≤i} log α_j,  γ_i = e^{lg_i},  Γ_ij = e^{lg_i − lg_j} (i ≥ j)
//! A  = strict_lower((K̂K̂ᵀ) ⊙ Γ),      L = A·diag(β)          (β_j scales COLUMN j)
//! P  = lower_incl((Q̂K̂ᵀ) ⊙ Γ)
//! R  = V − diag(γ) K̂ S_0
//! (I + L) E' = R,   U = diag(β) E'           (UT: T = (I+L)⁻¹ by forward substitution)
//! O  = diag(γ) Q̂ S_0 + P U
//! S_C = γ_{C−1} S_0 + K̂ᵀ diag(γ_{C−1}/γ_j) U
//! ```
//! Only the chunk-boundary states are kept (the same `states` layout as the
//! scan's checkpoints). The backward mirrors every line; the per-token decay
//! gradient is assembled from the mask/scale terms and a reverse cumsum
//! inside the chunk. Witness: `tests/vulkan_gdn_wy.rs` (production tile,
//! ≤ 5e-3 rel against the f64 token reference `ops::gdn_scan_ref_*`).
#![cfg(all(feature = "vulkan", not(target_os = "macos")))]

use crate::metal::{Cmd, Ctx, GBuf, GdnScanDims, GemmBatch, Op};

/// Chunk length of the WY form (the checkpoint stride of the scan).
pub const WY_C: usize = 64;

/// Forward intermediates of ONE GDN layer that its backward re-reads
/// (P, T, L, U, E', R, γ, β, q̂, k̂ …): allocated per layer — a shared arena
/// would hand the earlier layers' backward the last layer's tables.
pub struct GdnWyKeep {
    pub qn: GBuf,    // [M, nv·dk] q̂
    pub kn: GBuf,    // [M, nv·dk] k̂
    pub qg: GBuf,    // [M, nv·dk] γ·q̂
    pub kg: GBuf,    // [M, nv·dk] γ·k̂
    pub vcm: GBuf,   // [NB, nch, C, dv] V, then R = V − X0 (in place)
    pub epm: GBuf,   // [NB, nch, C, dv] E'
    pub um: GBuf,    // [NB, nch, C, dv] U
    pub utm: GBuf,   // [NB, nch, C, dv] Ũ = diag(ρ) U
    pub kk: GBuf,    // [NC, C, C] K̂K̂ᵀ (raw)
    pub qk: GBuf,    // [NC, C, C] P (masked)
    pub lmat: GBuf,  // [NC, C, C] L
    pub tmat: GBuf,  // [NC, C, C] T = (I+L)⁻¹
    pub lg: GBuf,    // [NB, T] cumulative log decay inside the chunk
    pub gam: GBuf,   // [NB, T] γ
    pub bet: GBuf,   // [NB, T] β
    pub rho: GBuf,   // [NB, T] γ_{C−1}/γ_j
}

/// Backward-only scratch of the WY form, shared by all GDN layers (every
/// buffer is fully rewritten inside one layer's backward).
pub struct GdnWyScratch {
    pub du: GBuf,    // [NB, C, dv]
    pub dut: GBuf,   // [NB, C, dv] K̂·dS_{c+1}
    pub dep: GBuf,   // [NB, C, dv] dE'
    pub dt: GBuf,    // [NB, C, C]
    pub dl1: GBuf,   // [NB, C, C]
    pub dp: GBuf,    // [NC, C, C] dP (then dP⊙Γ)
    pub dl: GBuf,    // [NC, C, C] dL (then dKK)
    pub dsn: GBuf,   // [NB, dk, dv] dS_c being assembled
    pub dqn: GBuf,   // [M, nv·dk]
    pub dkn: GBuf,   // [M, nv·dk]
    pub dqg: GBuf,   // [M, nv·dk] dQγ
    pub dkg: GBuf,   // [M, nv·dk] dKγ
    pub drho: GBuf,  // [NB, T]
    pub dbet: GBuf,  // [NB, T]
    pub dlgm: GBuf,  // [NB, T] mask contribution to d lg
    pub dgend: GBuf, // [NB, nch] Σ dS_{c+1}·S_c
    pub dla: GBuf,   // [NB, T] d log α
}

impl GdnWyKeep {
    pub fn new(c: &Ctx, d: &GdnScanDims) -> GdnWyKeep {
        assert!(d.t % WY_C == 0, "WY form needs T % 64 == 0");
        let m = d.rows();
        let nb = d.b * d.nv;
        let nch = d.t / WY_C;
        let nc = nb * nch;
        let z = |n: usize| GBuf::zeros(c, n);
        GdnWyKeep {
            qn: z(m * d.nv * d.dk),
            kn: z(m * d.nv * d.dk),
            qg: z(m * d.nv * d.dk),
            kg: z(m * d.nv * d.dk),
            vcm: z(nb * nch * WY_C * d.dv),
            epm: z(nb * nch * WY_C * d.dv),
            um: z(nb * nch * WY_C * d.dv),
            utm: z(nb * nch * WY_C * d.dv),
            kk: z(nc * WY_C * WY_C),
            qk: z(nc * WY_C * WY_C),
            lmat: z(nc * WY_C * WY_C),
            tmat: z(nc * WY_C * WY_C),
            lg: z(nb * d.t),
            gam: z(nb * d.t),
            bet: z(nb * d.t),
            rho: z(nb * d.t),
        }
    }
}

impl GdnWyScratch {
    pub fn new(c: &Ctx, d: &GdnScanDims) -> GdnWyScratch {
        assert!(d.t % WY_C == 0, "WY form needs T % 64 == 0");
        let m = d.rows();
        let nb = d.b * d.nv;
        let nch = d.t / WY_C;
        let nc = nb * nch;
        let z = |n: usize| GBuf::zeros(c, n);
        GdnWyScratch {
            du: z(nb * WY_C * d.dv),
            dut: z(nb * WY_C * d.dv),
            dep: z(nb * WY_C * d.dv),
            dt: z(nb * WY_C * WY_C),
            dl1: z(nb * WY_C * WY_C),
            dp: z(nc * WY_C * WY_C),
            dl: z(nc * WY_C * WY_C),
            dsn: z(nb * d.dk * d.dv),
            dqn: z(m * d.nv * d.dk),
            dkn: z(m * d.nv * d.dk),
            dqg: z(m * d.nv * d.dk),
            dkg: z(m * d.nv * d.dk),
            drho: z(nb * d.t),
            dbet: z(nb * d.t),
            dlgm: z(nb * d.t),
            dgend: z(nb * nch),
            dla: z(nb * d.t),
        }
    }
}

struct Geo {
    b: usize,
    t: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    nch: usize,
    ss: usize,
}

impl Geo {
    fn of(d: &GdnScanDims) -> Geo {
        Geo {
            b: d.b,
            t: d.t,
            nv: d.nv,
            dk: d.dk,
            dv: d.dv,
            nch: d.t / WY_C,
            ss: d.dk * d.dv,
        }
    }
    /// batch over (b, h, chunk) of a [M, nv·w]-layout token matrix
    fn tok3(&self, w: usize) -> [usize; 3] {
        [self.t * self.nv * w, w, WY_C * self.nv * w]
    }
    /// batch over (b, h) of a [M, nv·w]-layout token matrix, chunk c fixed
    fn tok2(&self, w: usize) -> [usize; 3] {
        [self.t * self.nv * w, w, 0]
    }
    /// batch of a [NC, C, C] chunk matrix over (b, h, chunk)
    fn cc3(&self) -> [usize; 3] {
        [self.nv * self.nch * WY_C * WY_C, self.nch * WY_C * WY_C, WY_C * WY_C]
    }
    fn cc2(&self) -> [usize; 3] {
        [self.nv * self.nch * WY_C * WY_C, self.nch * WY_C * WY_C, 0]
    }
    /// batch of a [NB, nch, C, dv] chunk-major value matrix over (b, h, chunk)
    fn cm3(&self) -> [usize; 3] {
        [self.nv * self.nch * WY_C * self.dv, self.nch * WY_C * self.dv, WY_C * self.dv]
    }
    fn cm2(&self) -> [usize; 3] {
        [self.nv * self.nch * WY_C * self.dv, self.nch * WY_C * self.dv, 0]
    }
    /// batch of the [NB, nch+1, dk, dv] state checkpoints over (b, h, chunk)
    fn st3(&self) -> [usize; 3] {
        [self.nv * (self.nch + 1) * self.ss, (self.nch + 1) * self.ss, self.ss]
    }
    fn st2(&self) -> [usize; 3] {
        [self.nv * (self.nch + 1) * self.ss, (self.nch + 1) * self.ss, 0]
    }
    /// batch of a [NB, C, x] per-chunk scratch over (b, h)
    fn sc2(&self, x: usize) -> [usize; 3] {
        [self.nv * WY_C * x, WY_C * x, 0]
    }
    fn st_batch(&self) -> [usize; 3] {
        [self.nv * self.ss, self.ss, 0]
    }
    fn bt3(&self, sa: [usize; 3], sb: [usize; 3], sc: [usize; 3]) -> GemmBatch {
        GemmBatch {
            nb: self.b,
            nh: self.nv,
            nc: self.nch,
            sa,
            sb,
            sc,
        }
    }
    fn bt2(&self, sa: [usize; 3], sb: [usize; 3], sc: [usize; 3]) -> GemmBatch {
        GemmBatch {
            nb: self.b,
            nh: self.nv,
            nc: 1,
            sa,
            sb,
            sc,
        }
    }
}

/// WY forward: same contract as `Cmd::gdn_scan_fwd` (`states` slot 0 = S_0
/// when `s0_from_ckpt`, zero otherwise; every chunk boundary state is left
/// in `states`; `raw_o` = Sᵀq̂ per token).
#[allow(clippy::too_many_arguments)]
pub fn wy_fwd(
    cmd: &Cmd,
    d: &GdnScanDims,
    w: &GdnWyKeep,
    qkv_cv: &GBuf,
    a_pre: &GBuf,
    b_pre: &GBuf,
    p: &GBuf,
    alog_off: usize,
    dt_off: usize,
    raw_o: &GBuf,
    states: &GBuf,
    s0_from_ckpt: bool,
    beta_one: bool,
) {
    let g = Geo::of(d);
    let (nv, dk, dv, nch, ss) = (g.nv, g.dk, g.dv, g.nch, g.ss);
    let nb = g.b * nv;
    if !s0_from_ckpt {
        // S_0 = 0 in checkpoint slot 0 of every (b, h)
        cmd.gdn_wy_zero_s0(states, nb, nch + 1, ss);
    }
    cmd.gdn_wy_scal(d, a_pre, b_pre, p, alog_off, dt_off, &w.lg, &w.gam, &w.bet, &w.rho, beta_one);
    cmd.gdn_wy_norm(d, qkv_cv, &w.gam, &w.qn, &w.kn, &w.qg, &w.kg, &w.vcm);
    // KK = K̂K̂ᵀ, QK = Q̂K̂ᵀ (lower-incl epilogue), batched over (b, h, c)
    cmd.gemm_ex(
        Op::N, Op::T, WY_C, WY_C, dk, 1.0, &w.kn, 0, nv * dk, &w.kn, 0, nv * dk, 0.0, &w.kk, 0,
        WY_C, &g.bt3(g.tok3(dk), g.tok3(dk), g.cc3()), false,
    );
    cmd.gemm_ex(
        Op::N, Op::T, WY_C, WY_C, dk, 1.0, &w.qn, 0, nv * dk, &w.kn, 0, nv * dk, 0.0, &w.qk, 0,
        WY_C, &g.bt3(g.tok3(dk), g.tok3(dk), g.cc3()), true,
    );
    cmd.gdn_wy_mask(d, &w.kk, &w.qk, &w.lg, &w.bet, &w.lmat);
    cmd.gdn_wy_ut(d, &w.lmat, &w.tmat);
    for c in 0..nch {
        // S_{c+1} := γ_{C−1}·S_c (the GEMM below adds K̂ᵀŨ)
        cmd.gdn_wy_sscale(d, states, &w.gam, c);
        // R_c = V_c − diag(γ)K̂_c S_c   (in place in vcm)
        cmd.gemm_ex(
            Op::N, Op::N, WY_C, dv, dk, -1.0, &w.kg, c * WY_C * nv * dk, nv * dk, states, c * ss,
            dv, 1.0, &w.vcm, c * WY_C * dv, dv, &g.bt2(g.tok2(dk), g.st2(), g.cm2()), false,
        );
        // E'_c = T_c R_c
        cmd.gemm_ex(
            Op::N, Op::N, WY_C, dv, WY_C, 1.0, &w.tmat, c * WY_C * WY_C, WY_C, &w.vcm,
            c * WY_C * dv, dv, 0.0, &w.epm, c * WY_C * dv, dv, &g.bt2(g.cc2(), g.cm2(), g.cm2()),
            false,
        );
        // U = diag(β)E', Ũ = diag(ρ)U
        cmd.gdn_wy_u(d, &w.epm, &w.bet, &w.rho, &w.um, &w.utm, c);
        // S_{c+1} += K̂_cᵀ Ũ_c
        cmd.gemm_ex(
            Op::T, Op::N, dk, dv, WY_C, 1.0, &w.kn, c * WY_C * nv * dk, nv * dk, &w.utm,
            c * WY_C * dv, dv, 1.0, states, (c + 1) * ss, dv, &g.bt2(g.tok2(dk), g.cm2(), g.st2()),
            false,
        );
    }
    // O = diag(γ)Q̂ S_c + P U   (batched over (b, h, c))
    cmd.gemm_ex(
        Op::N, Op::N, WY_C, dv, dk, 1.0, &w.qg, 0, nv * dk, states, 0, dv, 0.0, raw_o, 0, nv * dv,
        &g.bt3(g.tok3(dk), g.st3(), g.tok3(dv)), false,
    );
    cmd.gemm_ex(
        Op::N, Op::N, WY_C, dv, WY_C, 1.0, &w.qk, 0, WY_C, &w.um, 0, dv, 1.0, raw_o, 0, nv * dv,
        &g.bt3(g.cc3(), g.cm3(), g.tok3(dv)), false,
    );
}

/// WY backward: same contract as `Cmd::gdn_scan_bwd` (needs the forward's
/// scratch and `states`); `dlive` holds dS_T on entry when `ds_init` and
/// dS_0 on exit.
#[allow(clippy::too_many_arguments)]
pub fn wy_bwd(
    cmd: &Cmd,
    d: &GdnScanDims,
    w: &GdnWyKeep,
    x: &GdnWyScratch,
    qkv_cv: &GBuf,
    a_pre: &GBuf,
    p: &GBuf,
    alog_off: usize,
    dt_off: usize,
    states: &GBuf,
    doo: &GBuf,
    dlive: &GBuf,
    ds_init: bool,
    beta_one: bool,
    dcv: &GBuf,
    da: &GBuf,
    db: &GBuf,
    part: &GBuf,
) {
    let g = Geo::of(d);
    let (nv, dk, dv, nch, ss) = (g.nv, g.dk, g.dv, g.nch, g.ss);
    let nb = g.b * nv;
    let c_dim = d.c_dim;
    if !ds_init {
        cmd.axpby(0.0, dlive, 0.0, dlive, nb * ss);
    }
    for c in (0..nch).rev() {
        // dU = Pᵀ dO_c
        cmd.gemm_ex(
            Op::T, Op::N, WY_C, dv, WY_C, 1.0, &w.qk, c * WY_C * WY_C, WY_C, doo,
            c * WY_C * nv * dv, nv * dv, 0.0, &x.du, 0, dv, &g.bt2(g.cc2(), g.tok2(dv), g.sc2(dv)),
            false,
        );
        // dŨ = K̂_c dS_{c+1}   ([C×dk]·[dk×dv])
        cmd.gemm_ex(
            Op::N, Op::N, WY_C, dv, dk, 1.0, &w.kn, c * WY_C * nv * dk, nv * dk, dlive, 0, dv, 0.0,
            &x.dut, 0, dv, &g.bt2(g.tok2(dk), g.st_batch(), g.sc2(dv)), false,
        );
        // dU += ρ·dŨ ; dρ, dβ ; dE' = β dU
        cmd.gdn_wy_du(d, &x.du, &x.dut, &w.um, &w.epm, &w.bet, &w.rho, &x.dep, &x.drho, &x.dbet, c);
        // dP_c = dO_c U_cᵀ
        cmd.gemm_ex(
            Op::N, Op::T, WY_C, WY_C, dv, 1.0, doo, c * WY_C * nv * dv, nv * dv, &w.um,
            c * WY_C * dv, dv, 0.0, &x.dp, c * WY_C * WY_C, WY_C, &g.bt2(g.tok2(dv), g.cm2(), g.cc2()),
            false,
        );
        // dS_c = Qγ_cᵀ dO_c
        cmd.gemm_ex(
            Op::T, Op::N, dk, dv, WY_C, 1.0, &w.qg, c * WY_C * nv * dk, nv * dk, doo,
            c * WY_C * nv * dv, nv * dv, 0.0, &x.dsn, 0, dv, &g.bt2(g.tok2(dk), g.tok2(dv), g.st_batch()),
            false,
        );
        // dγ_end partial = Σ dS_{c+1}·S_c ; dS_c += γ_end dS_{c+1}
        cmd.gdn_wy_dgend(d, dlive, states, &x.dgend, c);
        cmd.gdn_wy_dstate(d, &x.dsn, dlive, &w.gam, c);
        // dK̂_c (from S_{c+1}) = Ũ_c dS_{c+1}ᵀ   (first write of the chunk rows)
        cmd.gemm_ex(
            Op::N, Op::T, WY_C, dk, dv, 1.0, &w.utm, c * WY_C * dv, dv, dlive, 0, dv, 0.0, &x.dkn,
            c * WY_C * nv * dk, nv * dk, &g.bt2(g.cm2(), g.st_batch(), g.tok2(dk)), false,
        );
        // dR = T_cᵀ dE'  → straight into the v columns of dcv
        cmd.gemm_ex(
            Op::T, Op::N, WY_C, dv, WY_C, 1.0, &w.tmat, c * WY_C * WY_C, WY_C, &x.dep, 0, dv, 0.0,
            dcv, c * WY_C * c_dim + 2 * nv * dk, c_dim, &g.bt2(g.cc2(), g.sc2(dv), [g.t * c_dim, dv, 0]),
            false,
        );
        // dT = dE' Rᵀ ; dL = −Tᵀ dT Tᵀ
        cmd.gemm_ex(
            Op::N, Op::T, WY_C, WY_C, dv, 1.0, &x.dep, 0, dv, &w.vcm, c * WY_C * dv, dv, 0.0, &x.dt,
            0, WY_C, &g.bt2(g.sc2(dv), g.cm2(), g.sc2(WY_C)), false,
        );
        cmd.gemm_ex(
            Op::T, Op::N, WY_C, WY_C, WY_C, 1.0, &w.tmat, c * WY_C * WY_C, WY_C, &x.dt, 0, WY_C, 0.0,
            &x.dl1, 0, WY_C, &g.bt2(g.cc2(), g.sc2(WY_C), g.sc2(WY_C)), false,
        );
        cmd.gemm_ex(
            Op::N, Op::T, WY_C, WY_C, WY_C, -1.0, &x.dl1, 0, WY_C, &w.tmat, c * WY_C * WY_C, WY_C,
            0.0, &x.dl, c * WY_C * WY_C, WY_C, &g.bt2(g.sc2(WY_C), g.cc2(), g.cc2()), false,
        );
        // dX0 = −dR: dS_c += −Kγ_cᵀ dR ; dKγ_c = −dR S_cᵀ
        cmd.gemm_ex(
            Op::T, Op::N, dk, dv, WY_C, -1.0, &w.kg, c * WY_C * nv * dk, nv * dk, dcv,
            c * WY_C * c_dim + 2 * nv * dk, c_dim, 1.0, &x.dsn, 0, dv,
            &g.bt2(g.tok2(dk), [g.t * c_dim, dv, 0], g.st_batch()), false,
        );
        cmd.gemm_ex(
            Op::N, Op::T, WY_C, dk, dv, -1.0, dcv, c * WY_C * c_dim + 2 * nv * dk, c_dim, states,
            c * ss, dv, 0.0, &x.dkg, c * WY_C * nv * dk, nv * dk,
            &g.bt2([g.t * c_dim, dv, 0], g.st2(), g.tok2(dk)), false,
        );
        // carry dS_c
        cmd.copy(&x.dsn, 0, dlive, 0, nb * ss);
    }
    // decay/β terms of the masks, then the masks' gradients scaled by Γ (and β_j)
    cmd.gdn_wy_mask_bwd_red(d, &x.dp, &w.qk, &x.dl, &w.lmat, &w.kk, &w.lg, &w.bet, &x.dlgm, &x.dbet);
    cmd.gdn_wy_mask_bwd(d, &x.dp, &x.dl, &w.lg, &w.bet);
    // dQ̂ = dPg K̂ ; dK̂ += dPgᵀ Q̂ + dKKg K̂ + dKKgᵀ K̂ ; dQγ = dO S_cᵀ   (batched over (b, h, c))
    cmd.gemm_ex(
        Op::N, Op::N, WY_C, dk, WY_C, 1.0, &x.dp, 0, WY_C, &w.kn, 0, nv * dk, 0.0, &x.dqn, 0, nv * dk,
        &g.bt3(g.cc3(), g.tok3(dk), g.tok3(dk)), false,
    );
    cmd.gemm_ex(
        Op::T, Op::N, WY_C, dk, WY_C, 1.0, &x.dp, 0, WY_C, &w.qn, 0, nv * dk, 1.0, &x.dkn, 0, nv * dk,
        &g.bt3(g.cc3(), g.tok3(dk), g.tok3(dk)), false,
    );
    cmd.gemm_ex(
        Op::N, Op::N, WY_C, dk, WY_C, 1.0, &x.dl, 0, WY_C, &w.kn, 0, nv * dk, 1.0, &x.dkn, 0, nv * dk,
        &g.bt3(g.cc3(), g.tok3(dk), g.tok3(dk)), false,
    );
    cmd.gemm_ex(
        Op::T, Op::N, WY_C, dk, WY_C, 1.0, &x.dl, 0, WY_C, &w.kn, 0, nv * dk, 1.0, &x.dkn, 0, nv * dk,
        &g.bt3(g.cc3(), g.tok3(dk), g.tok3(dk)), false,
    );
    cmd.gemm_ex(
        Op::N, Op::T, WY_C, dk, dv, 1.0, doo, 0, nv * dv, states, 0, dv, 0.0, &x.dqg, 0, nv * dk,
        &g.bt3(g.tok3(dv), g.st3(), g.tok3(dk)), false,
    );
    // per-token decay gradient (reverse cumsum inside the chunk), then da/db/dA_log/ddt and the q/k norm backward
    cmd.gdn_wy_dlg(d, &x.dqn, &x.dkn, &x.dqg, &x.dkg, &w.qn, &w.kn, &x.drho, &x.dlgm, &x.dgend, &w.rho, &w.gam, &x.dla);
    cmd.gdn_wy_dscal(d, &x.dla, a_pre, p, alog_off, dt_off, &x.dbet, &w.bet, da, db, part, beta_one);
    cmd.gdn_wy_dnorm(d, &x.dqn, &x.dkn, &x.dqg, &x.dkg, qkv_cv, &w.gam, dcv);
}
