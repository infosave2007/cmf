//! The DFlash 2 block on CUDA, beside the hybrid arm whose kernels it uses.
//!
//! The sidecar's four-bit matrices (q, k, v, o, the gate/up pair, down) go
//! through the hybrid arm's tensor-core matvec, repacked on the card; the
//! eight-bit ones (fc, the kernel projections, the selector's projection)
//! are f16 on the card for cuBLAS. The row convolution, the head norms with
//! the rotation and the attention are `dflash.cu`. The context's keys and
//! values live here per layer (`[position − first][kv][hd]`, f32), made on
//! the card from the target's tap rows (`settle`). A block returns its draft
//! rows' logits over the head's first `width` rows and the selector's
//! projection of the same rows; the host picks the tokens
//! (`dflash::Drafter::select_with`).

use super::*;

const SRC: &str = include_str!("../dflash.cu");
/// Context rows a settle takes through the tensor-core matvec at a time.
const SETTLE_ROWS: usize = NMAX;
/// Context rows one settle takes (fc's GEMM reads its matrix once for them).
pub const SETTLE_MAX: usize = 256;

/// What a layer of the sidecar hands the card.
pub struct LayerIn<'a> {
    pub in_norm: &'a [f32],
    pub post_norm: &'a [f32],
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
    pub attn_base: &'a [f32],
    pub mlp_base: &'a [f32],
    /// The kernel projections, f32 `[2 kn groups][hs]`.
    pub attn_proj: &'a [f32],
    pub mlp_proj: &'a [f32],
    pub window: usize,
    pub rotary: bool,
    pub q: GraphW<'a>,
    pub k: GraphW<'a>,
    pub v: GraphW<'a>,
    pub o: GraphW<'a>,
    pub gate: GraphW<'a>,
    pub up: GraphW<'a>,
    pub down: GraphW<'a>,
}

/// The sidecar's geometry and its matrices for the card.
pub struct SpecIn<'a> {
    pub hs: usize,
    pub ffn: usize,
    pub hd: usize,
    pub nh: usize,
    pub kv: usize,
    pub kn: usize,
    pub gs: usize,
    pub block: usize,
    pub eps: f32,
    pub sel_rank: usize,
    /// Joined tap width (taps × hs).
    pub tw: usize,
    pub layers: Vec<LayerIn<'a>>,
    pub norm: &'a [f32],
    pub hidden_norm: &'a [f32],
    pub inv_freq: &'a [f32],
    /// fc f32 `[hs][tw]`, the selector's projection f32 `[sel_rank][hs]`.
    pub fc: &'a [f32],
    pub sel: &'a [f32],
}

struct Fx {
    conv: cs::CUfunction,
    qk: cs::CUfunction,
    attn: cs::CUfunction,
}

struct Layer {
    in_norm: Buf,
    post_norm: Buf,
    q_norm: Buf,
    k_norm: Buf,
    attn_base: Buf,
    mlp_base: Buf,
    attn_proj: Buf,
    mlp_proj: Buf,
    window: usize,
    rotary: bool,
    qkv: MmaSet,
    o: MmaSet,
    gu: MmaSet,
    down: MmaSet,
    /// k and v of context rows (settle).
    kv: MmaSet,
    ctx_k: Buf,
    ctx_v: Buf,
    cap: usize,
}

/// The block's device state.
pub struct Dev {
    hs: usize,
    ffn: usize,
    hd: usize,
    nh: usize,
    kv: usize,
    kn: usize,
    gs: usize,
    block: usize,
    eps: f32,
    rank: usize,
    tw: usize,
    fx: Fx,
    hfx: Arc<HFuncs>,
    layers: Vec<Layer>,
    norm: Buf,
    hidden_norm: Buf,
    inv: Buf,
    fc: Buf,
    sel: Buf,
    // activations of one block (`block` rows) or one settle (64 rows)
    h: Buf,
    x: Buf,
    xc: Buf,
    d: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    y: Buf,
    o: Buf,
    x16: Buf,
    p: [Buf; 3],
    pstride: usize,
    mid: [Buf; 2],
    taps: Buf,
    t: Buf,
    kc: Buf,
    vc: Buf,
    /// A settle's keys and values of every layer, `[layer][k | v][64][kv hd]`,
    /// read back in one copy.
    kvall: Buf,
    hp: Buf,
    logits: Buf,
    logits_cap: usize,
    /// The head's job table: (model, tensor, width) it was made for.
    head: Option<((u64, usize, usize), MmaSet)>,
}
unsafe impl Send for Dev {}

fn f16s(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| cortiq_core::quant::f32_to_f16(x)).collect()
}

impl Dev {
    /// The sidecar's matrices onto the card. Needs the hybrid arm's kernels
    /// (a trunk plan built first).
    pub fn new(model: &Arc<CmfModel>, s: &SpecIn) -> Res<Self> {
        let c = super::super::ctx().ok_or("no CUDA context")?;
        let mut st = c.st.lock().unwrap();
        c.current()?;
        let st = &mut *st;
        let hfx = st.hyb.funcs.values().next().cloned().ok_or("the hybrid kernels are not built yet")?;
        let module = compile(SRC, Vec::new(), c.arch)?;
        let fx = Fx { conv: func(module, "df_conv")?, qk: func(module, "df_qk")?, attn: func(module, "df_attn")? };
        let (hs, ffn, hd, nh, kv) = (s.hs, s.ffn, s.hd, s.nh, s.kv);
        if hs % 32 != 0 || hd > 256 || nh % kv != 0 || s.block > NMAX || s.fc.len() != hs * s.tw || s.sel.len() != s.sel_rank * hs {
            return Err("a DFlash geometry the CUDA block does not take".into());
        }
        let rows = SETTLE_MAX.max(s.block);
        let pstride = hs.max(nh * hd);
        let p = [Buf::new(rows * pstride * 2)?, Buf::new(rows * pstride * 2)?, Buf::new(rows * pstride * 2)?];
        let mid = [Buf::new(NMAX * ffn * 2)?, Buf::new(NMAX * ffn * 2)?];
        let (q, k, v) = (Buf::new(NMAX * nh * hd * 4)?, Buf::new(NMAX * kv * hd * 4)?, Buf::new(NMAX * kv * hd * 4)?);
        let o = Buf::new(NMAX * hs * 4)?;
        let (kc, vc) = (Buf::new(SETTLE_ROWS * kv * hd * 4)?, Buf::new(SETTLE_ROWS * kv * hd * 4)?);
        let mut layers = Vec::with_capacity(s.layers.len());
        for l in &s.layers {
            let qm = hmat(c, st, model, &l.q, nh * hd, hs)?;
            let km = hmat(c, st, model, &l.k, kv * hd, hs)?;
            let vm = hmat(c, st, model, &l.v, kv * hd, hs)?;
            let om = hmat(c, st, model, &l.o, hs, nh * hd)?;
            let gum = hmat_pair(c, st, model, &l.gate, &l.up, ffn, hs)?;
            let dm = hmat(c, st, model, &l.down, hs, ffn)?;
            layers.push(Layer {
                in_norm: Buf::from_slice(l.in_norm)?,
                post_norm: Buf::from_slice(l.post_norm)?,
                q_norm: Buf::from_slice(l.q_norm)?,
                k_norm: Buf::from_slice(l.k_norm)?,
                attn_base: Buf::from_slice(l.attn_base)?,
                mlp_base: Buf::from_slice(l.mlp_base)?,
                attn_proj: Buf::from_slice(&f16s(l.attn_proj))?,
                mlp_proj: Buf::from_slice(&f16s(l.mlp_proj))?,
                window: l.window,
                rotary: l.rotary,
                qkv: mmaset(&[(&qm, q.ptr, 0, nh * hd, nh * hd), (&km, k.ptr, 0, kv * hd, kv * hd), (&vm, v.ptr, 0, kv * hd, kv * hd)])?,
                o: mmaset(&[(&om, o.ptr, 0, hs, hs)])?,
                gu: mmaset_pair(&gum, mid[0].ptr, mid[1].ptr, ffn)?,
                down: mmaset(&[(&dm, o.ptr, 0, hs, hs)])?,
                kv: mmaset(&[(&km, kc.ptr, 0, kv * hd, kv * hd), (&vm, vc.ptr, 0, kv * hd, kv * hd)])?,
                ctx_k: Buf::new(64)?,
                ctx_v: Buf::new(64)?,
                cap: 0,
            });
        }
        Ok(Self {
            hs,
            ffn,
            hd,
            nh,
            kv,
            kn: s.kn,
            gs: s.gs,
            block: s.block,
            eps: s.eps,
            rank: s.sel_rank,
            tw: s.tw,
            fx,
            hfx,
            layers,
            norm: Buf::from_slice(s.norm)?,
            hidden_norm: Buf::from_slice(s.hidden_norm)?,
            inv: Buf::from_slice(s.inv_freq)?,
            fc: Buf::from_slice(&f16s(s.fc))?,
            sel: Buf::from_slice(&f16s(s.sel))?,
            h: Buf::new(rows * hs * 4)?,
            x: Buf::new(rows * hs * 4)?,
            xc: Buf::new(rows * hs * 4)?,
            d: Buf::new(rows * 2 * s.kn * (hs / s.gs) * 4)?,
            q,
            k,
            v,
            y: Buf::new(NMAX * nh * hd * 4)?,
            o,
            x16: Buf::new(rows * s.tw.max(hs) * 2)?,
            p,
            pstride,
            mid,
            taps: Buf::new(rows * s.tw * 4)?,
            t: Buf::new(rows * hs * 4)?,
            kc,
            vc,
            kvall: Buf::new(s.layers.len() * 2 * SETTLE_MAX * kv * hd * 4)?,
            hp: Buf::new(NMAX * s.sel_rank * 4)?,
            logits: Buf::new(64)?,
            logits_cap: 0,
            head: None,
        })
    }

    /// Room for `rows` context rows of a layer; what it held stays.
    pub fn reserve(&mut self, layer: usize, rows: usize) -> bool {
        let Some(c) = super::super::ctx() else { return false };
        let l = &mut self.layers[layer];
        if rows <= l.cap {
            return true;
        }
        if c.current().is_err() {
            return false;
        }
        let cap = rows.max(2 * l.cap).max(4096);
        let bytes = cap * self.kv * self.hd * 4;
        let (Ok(nk), Ok(nv)) = (Buf::new(bytes), Buf::new(bytes)) else { return false };
        let old = l.cap * self.kv * self.hd * 4;
        unsafe {
            if old > 0 {
                let _ = cu::memcpy_dtod_async(nk.ptr, l.ctx_k.ptr, old, c.stream);
                let _ = cu::memcpy_dtod_async(nv.ptr, l.ctx_v.ptr, old, c.stream);
            }
        }
        let _ = c.sync();
        (l.ctx_k, l.ctx_v, l.cap) = (nk, nv, cap);
        true
    }

    /// Host-made context rows of a layer at `at` (position − first).
    pub fn push_rows(&mut self, layer: usize, at: usize, k: &[f32], v: &[f32]) -> bool {
        let Some(c) = super::super::ctx() else { return false };
        let row = self.kv * self.hd;
        let n = k.len() / row;
        if v.len() != k.len() || k.len() % row != 0 || !self.reserve(layer, at + n) {
            return false;
        }
        let l = &self.layers[layer];
        let off = (at * row * 4) as u64;
        unsafe {
            cu::memcpy_htod_async(l.ctx_k.ptr + off, k, c.stream).is_ok() && cu::memcpy_htod_async(l.ctx_v.ptr + off, v, c.stream).is_ok() && c.sync().is_ok()
        }
    }

    /// `n` tap rows (`[n][tw]`) of positions `pos ..` into the context of
    /// every layer at `at` (position − first): t = norm(fc · taps), and per
    /// layer its keys (normed, rotated) and values. Returns each layer's
    /// keys and values for the host's copy.
    pub fn settle(&mut self, at: usize, pos: usize, taps: &[f32], n: usize) -> Res<Vec<(Vec<f32>, Vec<f32>)>> {
        let c = super::super::ctx().ok_or("no CUDA context")?;
        let (hs, tw, kv, hd) = (self.hs, self.tw, self.kv, self.hd);
        if n == 0 || n > SETTLE_MAX || taps.len() != n * tw {
            return Err("a settle of 1 to 256 tap rows".into());
        }
        for li in 0..self.layers.len() {
            if !self.reserve(li, at + n) {
                return Err("context rows do not fit the card".into());
            }
        }
        let _st = c.st.lock().unwrap();
        c.current()?;
        let t0 = std::time::Instant::now();
        let fx = &self.hfx;
        let row = kv * hd;
        let mut out: Vec<(Vec<f32>, Vec<f32>)> = (0..self.layers.len()).map(|_| (vec![0f32; n * row], vec![0f32; n * row])).collect();
        let (hs_i, eps, zero_i) = (hs as i32, self.eps, 0i32);
        let nul = 0u64;
        unsafe {
            cu::memcpy_htod_async(self.taps.ptr, taps, c.stream).map_err(derr)?;
            let tot = (n * tw) as u64;
            c.launch(fx.to_half, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(self.taps.ptr, self.x16.ptr, tot))?;
            c.gemm(true, false, hs, n, tw, 1.0, self.fc.ptr, tw, self.x16.ptr, tw, 0.0, self.t.ptr, hs, Gemm::Half)?;
            c.launch(fx.rmsnorm2, (hs.div_ceil(1024) as u32, n as u32), 256, 0, kargs!(self.t.ptr, self.hidden_norm.ptr, self.x.ptr, hs_i, eps, zero_i, nul, nul, nul, zero_i))?;
            let planes = Planes { p0: self.p[0].ptr, p1: self.p[1].ptr, p2: self.p[2].ptr, stride: self.pstride };
            split(c, fx, &planes, self.x.ptr, hs, hs, n)?;
            let kvh = kv as i32;
            let hd_i = hd as i32;
            for (li, l) in self.layers.iter().enumerate() {
                let mut done = 0;
                while done < n {
                    let m = (n - done).min(SETTLE_ROWS);
                    let sub = Planes { p0: planes.p0 + (done * self.pstride * 2) as u64, p1: planes.p1 + (done * self.pstride * 2) as u64, p2: planes.p2 + (done * self.pstride * 2) as u64, stride: self.pstride };
                    l.kv.run(c, fx, &sub, hs, m)?;
                    let (p0_i, rot_i) = ((pos + done) as i32, l.rotary as i32);
                    c.launch(self.fx.qk, (kv as u32, m as u32), hd as u32, 0, kargs!(self.kc.ptr, l.k_norm.ptr, self.inv.ptr, kvh, hd_i, p0_i, rot_i, eps))?;
                    let off = ((at + done) * row * 4) as u64;
                    cu::memcpy_dtod_async(l.ctx_k.ptr + off, self.kc.ptr, m * row * 4, c.stream).map_err(derr)?;
                    cu::memcpy_dtod_async(l.ctx_v.ptr + off, self.vc.ptr, m * row * 4, c.stream).map_err(derr)?;
                    let st_k = self.kvall.ptr + (((li * 2) * SETTLE_MAX + done) * row * 4) as u64;
                    let st_v = self.kvall.ptr + (((li * 2 + 1) * SETTLE_MAX + done) * row * 4) as u64;
                    cu::memcpy_dtod_async(st_k, self.kc.ptr, m * row * 4, c.stream).map_err(derr)?;
                    cu::memcpy_dtod_async(st_v, self.vc.ptr, m * row * 4, c.stream).map_err(derr)?;
                    done += m;
                }
            }
        }
        // every layer's keys and values back at once
        let mut all = vec![0f32; self.layers.len() * 2 * SETTLE_MAX * row];
        unsafe { cu::memcpy_dtoh_async(all.as_mut_slice(), self.kvall.ptr, c.stream).map_err(derr)? };
        c.sync()?;
        for (li, (k, v)) in out.iter_mut().enumerate() {
            k.copy_from_slice(&all[(li * 2) * SETTLE_MAX * row..][..n * row]);
            v.copy_from_slice(&all[(li * 2 + 1) * SETTLE_MAX * row..][..n * row]);
        }
        if debug() {
            eprintln!("dflash settle: {n} rows at {at} in {:.2} ms", t0.elapsed().as_secs_f64() * 1e3);
        }
        Ok(out)
    }

    /// One block: the rows `[anchor, MASK…]` (`h`, their embeddings) at
    /// positions `anchor ..`, layer `li` reading its context rows
    /// `ranges[li]` = (index into the context, rows, position of the first).
    /// Returns the logits of rows 1..=k over the head's first `width` rows
    /// and the selector's projection of those rows.
    #[allow(clippy::too_many_arguments)]
    pub fn block(
        &mut self,
        anchor: usize,
        h: &[f32],
        ranges: &[(usize, usize, usize)],
        head_model: &Arc<CmfModel>,
        head: &GraphW,
        head_rows: usize,
        width: usize,
        k: usize,
    ) -> Res<(Vec<f32>, Vec<f32>)> {
        let c = super::super::ctx().ok_or("no CUDA context")?;
        let (hs, hd, nh, kv, rows) = (self.hs, self.hd, self.nh, self.kv, self.block);
        if h.len() != rows * hs || ranges.len() != self.layers.len() || k == 0 || k >= rows || width == 0 || width > head_rows {
            return Err("a DFlash block of the sidecar's rows".into());
        }
        if ranges.iter().any(|&(_, n, _)| (n + rows) * 4 > 44 * 1024) {
            return Err("a context longer than the attention's scores take".into());
        }
        let mut st = c.st.lock().unwrap();
        c.current()?;
        let st = &mut *st;
        let hkey = (head_model.uid(), head.idx, width);
        if self.head.as_ref().is_none_or(|(kk, _)| *kk != hkey) || self.logits_cap < NMAX * width {
            if self.logits_cap < NMAX * width {
                self.logits = Buf::new(NMAX * width * 4)?;
                self.logits_cap = NMAX * width;
            }
            let m = hmat(c, st, head_model, head, head_rows, hs)?;
            self.head = Some((hkey, mmaset(&[(&m, self.logits.ptr, 0, width, width)])?));
        }
        let fx = &self.hfx;
        let (hs_i, eps, zero_i, one_i) = (hs as i32, self.eps, 0i32, 1i32);
        let nul = 0u64;
        let (kn_i, gs_i, rows_i) = (self.kn as i32, self.gs as i32, rows as i32);
        let pr_i = (2 * self.kn * (hs / self.gs)) as i32;
        let planes = Planes { p0: self.p[0].ptr, p1: self.p[1].ptr, p2: self.p[2].ptr, stride: self.pstride };
        let mid = Planes { p0: self.mid[0].ptr, p1: self.mid[1].ptr, p2: self.mid[1].ptr, stride: self.ffn };
        let scale = (hd as f32).powf(-0.5);
        let grid_h = (hs.div_ceil(256) as u32, rows as u32);
        let (nh_i, kv_i, hd_i, anchor_i) = (nh as i32, kv as i32, hd as i32, anchor as i32);
        unsafe {
            cu::memcpy_htod_async(self.h.ptr, h, c.stream).map_err(derr)?;
            for (li, l) in self.layers.iter().enumerate() {
                // attention: x = norm(h); x, d = conv_in(x); q, k, v
                let pre = |nw: u64, proj: u64, base: u64| -> Res<()> {
                    c.launch(fx.rmsnorm2, (hs.div_ceil(1024) as u32, rows as u32), 256, 0, kargs!(self.h.ptr, nw, self.x.ptr, hs_i, eps, zero_i, nul, nul, nul, zero_i))?;
                    let tot = (rows * hs) as u64;
                    c.launch(fx.to_half, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(self.x.ptr, self.x16.ptr, tot))?;
                    let pr = 2 * self.kn * (hs / self.gs);
                    c.gemm(true, false, pr, rows, hs, 1.0, proj, hs, self.x16.ptr, hs, 0.0, self.d.ptr, pr, Gemm::Half)?;
                    let stage0 = 0i32;
                    c.launch(self.fx.conv, grid_h, 256, 0, kargs!(self.x.ptr, self.d.ptr, base, self.xc.ptr, hs_i, kn_i, gs_i, stage0, rows_i, zero_i, pr_i))?;
                    split(c, fx, &planes, self.xc.ptr, hs, hs, rows)
                };
                pre(l.in_norm.ptr, l.attn_proj.ptr, l.attn_base.ptr)?;
                l.qkv.run(c, fx, &planes, hs, rows)?;
                let (rot_i, a0) = (l.rotary as i32, anchor as i32);
                c.launch(self.fx.qk, (nh as u32, rows as u32), hd as u32, 0, kargs!(self.q.ptr, l.q_norm.ptr, self.inv.ptr, nh_i, hd_i, a0, rot_i, eps))?;
                c.launch(self.fx.qk, (kv as u32, rows as u32), hd as u32, 0, kargs!(self.k.ptr, l.k_norm.ptr, self.inv.ptr, kv_i, hd_i, a0, rot_i, eps))?;
                let (lo, n, lo_abs) = ranges[li];
                let (lo_i, n_i, la_i, w_i) = (lo as i32, n as i32, lo_abs as i32, l.window as i32);
                c.launch(self.fx.attn, (nh as u32, rows as u32), 128, (n + rows) * 4, kargs!(self.q.ptr, self.k.ptr, self.v.ptr, l.ctx_k.ptr, l.ctx_v.ptr, self.y.ptr, rows_i, nh_i, kv_i, hd_i, lo_i, n_i, la_i, anchor_i, w_i, scale))?;
                split(c, fx, &planes, self.y.ptr, nh * hd, nh * hd, rows)?;
                l.o.run(c, fx, &planes, nh * hd, rows)?;
                let stage1 = 1i32;
                c.launch(self.fx.conv, grid_h, 256, 0, kargs!(self.o.ptr, self.d.ptr, l.attn_base.ptr, self.h.ptr, hs_i, kn_i, gs_i, stage1, rows_i, one_i, pr_i))?;
                // the MLP the same way
                pre(l.post_norm.ptr, l.mlp_proj.ptr, l.mlp_base.ptr)?;
                l.gu.run(c, fx, &planes, hs, rows)?;
                l.down.run(c, fx, &mid, self.ffn, rows)?;
                c.launch(self.fx.conv, grid_h, 256, 0, kargs!(self.o.ptr, self.d.ptr, l.mlp_base.ptr, self.h.ptr, hs_i, kn_i, gs_i, stage1, rows_i, one_i, pr_i))?;
            }
            // the last norm; the head and the selector over rows 1..=k
            c.launch(fx.rmsnorm2, (hs.div_ceil(1024) as u32, rows as u32), 256, 0, kargs!(self.h.ptr, self.norm.ptr, self.x.ptr, hs_i, eps, zero_i, nul, nul, nul, zero_i))?;
            let row1 = self.x.ptr + (hs * 4) as u64;
            split(c, fx, &planes, row1, hs, hs, k)?;
            self.head.as_ref().unwrap().1.run(c, fx, &planes, hs, k)?;
            let tot = (k * hs) as u64;
            c.launch(fx.to_half, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(row1, self.x16.ptr, tot))?;
            c.gemm(true, false, self.rank, k, hs, 1.0, self.sel.ptr, hs, self.x16.ptr, hs, 0.0, self.hp.ptr, self.rank, Gemm::Half)?;
            let mut logits = vec![0f32; k * width];
            let mut hp = vec![0f32; k * self.rank];
            cu::memcpy_dtoh_async(logits.as_mut_slice(), self.logits.ptr, c.stream).map_err(derr)?;
            cu::memcpy_dtoh_async(hp.as_mut_slice(), self.hp.ptr, c.stream).map_err(derr)?;
            c.sync()?;
            Ok((logits, hp))
        }
    }
}
