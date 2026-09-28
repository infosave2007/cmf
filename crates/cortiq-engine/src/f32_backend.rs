//! Explicit, thread-local F32 execution policy. The historical scalar path is
//! the default. Optimized reductions are NOT bit-exact: callers must audit the
//! complete model before opting in. This changes runtime arithmetic, not CMF.
use std::cell::Cell;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Backend {
    #[default]
    Reference,
    Accelerate,
}

thread_local! {
    static BACKEND: Cell<Backend> = const { Cell::new(Backend::Reference) };
    static CALLS: Cell<u64> = const { Cell::new(0) };
}

pub fn available(backend: Backend) -> bool {
    backend == Backend::Reference || cfg!(target_os = "macos")
}

pub fn active() -> bool {
    BACKEND.with(|b| b.get() != Backend::Reference)
}

pub fn calls() -> u64 {
    CALLS.with(Cell::get)
}

/// Nested scopes and unwinding restore the previous policy; other threads are
/// unaffected. An unavailable explicit backend is an error, never a false label.
pub fn scope<T>(backend: Backend, f: impl FnOnce() -> T) -> T {
    assert!(available(backend), "requested F32 backend is unavailable");
    struct Restore(Backend);
    impl Drop for Restore {
        fn drop(&mut self) {
            BACKEND.with(|b| b.set(self.0));
        }
    }
    let _restore = Restore(BACKEND.with(|b| b.replace(backend)));
    f()
}

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemv(
        order: i32,
        trans: i32,
        m: i32,
        n: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        x: *const f32,
        incx: i32,
        beta: f32,
        y: *mut f32,
        incy: i32,
    );
    fn cblas_sgemm(
        order: i32,
        ta: i32,
        tb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

/// Historical F32 matvec permits a short output and uses x.len() as stride.
pub(crate) fn matvec(w: &[f32], x: &[f32], y: &mut [f32]) -> bool {
    if !active() {
        return false;
    }
    #[cfg(target_os = "macos")]
    {
        let (m, n) = (y.len(), x.len());
        assert!(
            m.checked_mul(n).is_some_and(|len| len <= w.len()),
            "short F32 weights"
        );
        if m == 0 {
            return true;
        }
        if n == 0 {
            y.fill(0.);
            return true;
        }
        let (Ok(m), Ok(n)) = (i32::try_from(m), i32::try_from(n)) else {
            return false;
        };
        // RowMajor=101, NoTrans=111, beta=0: all output cells are overwritten.
        unsafe {
            cblas_sgemv(
                101,
                111,
                m,
                n,
                1.,
                w.as_ptr(),
                n,
                x.as_ptr(),
                1,
                0.,
                y.as_mut_ptr(),
                1,
            );
        }
        CALLS.with(|c| c.set(c.get() + 1));
        return true;
    }
    #[allow(unreachable_code)]
    false
}

/// Input [batch, cols], weights [rows, cols], output [batch, rows].
pub(crate) fn matmat(
    w: &[f32],
    x: &[f32],
    batch: usize,
    rows: usize,
    cols: usize,
    y: &mut [f32],
) -> bool {
    if !active() {
        return false;
    }
    #[cfg(target_os = "macos")]
    {
        assert_eq!(batch.checked_mul(cols), Some(x.len()), "F32 input shape");
        assert_eq!(batch.checked_mul(rows), Some(y.len()), "F32 output shape");
        assert_eq!(rows.checked_mul(cols), Some(w.len()), "F32 weight shape");
        if batch == 0 || rows == 0 {
            return true;
        }
        if cols == 0 {
            y.fill(0.);
            return true;
        }
        let (Ok(m), Ok(n), Ok(k)) = (
            i32::try_from(batch),
            i32::try_from(rows),
            i32::try_from(cols),
        ) else {
            return false;
        };
        unsafe {
            cblas_sgemm(
                101,
                111,
                112,
                m,
                n,
                k,
                1.,
                x.as_ptr(),
                k,
                w.as_ptr(),
                k,
                0.,
                y.as_mut_ptr(),
                n,
            );
        }
        CALLS.with(|c| c.set(c.get() + 1));
        return true;
    }
    #[allow(unreachable_code)]
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reference_is_default_and_does_not_write() {
        let mut y = [7.];
        assert!(!matvec(&[2.], &[3.], &mut y));
        assert_eq!(y, [7.]);
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn scoped_blas_shapes_overwrite_and_restore() {
        scope(Backend::Accelerate, || {
            let w = [1., 2., 3., -1., 4., 2.];
            let x = [2., 3., 4.];
            let mut y = [f32::NAN; 2];
            assert!(matvec(&w, &x, &mut y));
            assert_eq!(y, [20., 18.]);
            assert!(matvec(&w, &x, &mut y[..1]));
            let mut ys = [f32::NAN; 4];
            assert!(matmat(&w, &[2., 3., 4., 1., 0., -1.], 2, 2, 3, &mut ys));
            assert_eq!(ys, [20., 18., -2., -3.]);
            scope(Backend::Reference, || assert!(!active()));
            assert!(active());
            std::thread::spawn(|| assert!(!active())).join().unwrap();
        });
        assert!(!active());
        let _ = std::panic::catch_unwind(|| scope(Backend::Accelerate, || panic!("restore")));
        assert!(!active());
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn qtensor_single_many_and_batch_match_reference() {
        use crate::{pool::Pool, qtensor::QTensor};
        let pool = Pool::new(2);
        for (rows, cols, batch) in [(1, 1, 1), (5, 7, 3), (300, 64, 7)] {
            let w: Vec<_> = (0..rows * cols)
                .map(|i| ((i * 7 % 97) as f32 - 48.) / 49.)
                .collect();
            let t = QTensor::from_f32(w, rows, cols);
            let xs: Vec<_> = (0..batch * cols).map(|i| (i as f32 * 0.17).sin()).collect();
            let mut reference = vec![0.; batch * rows];
            t.matmat(&xs, batch, &mut reference, Some(&pool));
            scope(Backend::Accelerate, || {
                let mut got = vec![f32::NAN; batch * rows];
                t.matmat(&xs, batch, &mut got, Some(&pool));
                for (a, b) in got.iter().zip(&reference) {
                    assert!((a - b).abs() < 2e-5);
                }
                let (mut a, mut b) = (vec![f32::NAN; rows], vec![f32::NAN; rows]);
                QTensor::matvec_many([&t, &t], &xs[..cols], [&mut a, &mut b], Some(&pool));
                assert_eq!(a, b);
                for (a, b) in a.iter().zip(&reference) {
                    assert!((a - b).abs() < 2e-5);
                }
            });
        }
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn empty_and_malformed_shapes_never_reach_ffi() {
        scope(Backend::Accelerate, || {
            let mut y = [f32::NAN; 3];
            assert!(matvec(&[], &[], &mut y));
            assert_eq!(y, [0.; 3]);
            assert!(matmat(&[], &[], 1, 3, 0, &mut y));
            assert_eq!(y, [0.; 3]);
            assert!(std::panic::catch_unwind(|| matvec(&[1.], &[1., 2.], &mut [0.])).is_err());
            assert!(
                std::panic::catch_unwind(|| matmat(&[1.], &[1.], 2, 1, 1, &mut [0.; 2])).is_err()
            );
        });
    }
}
