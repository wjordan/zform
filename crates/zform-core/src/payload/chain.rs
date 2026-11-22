//! Linear numeric chains (`ProgramI64` with one input) lowered to one in-place loop.

use super::ir::{Len, Op, Program, Width};

/// Lower a linear numeric SSA chain to one in-place loop over `buf`. The operations and constants
/// still come entirely from the payload; short columns avoid a handler transition and a second
/// allocation.
pub(crate) fn eval_numeric_chain_in_place(p: &Program, buf: &mut [i64]) -> bool {
    if p.width != Width::W64
        || p.n_in != 1
        || p.n_bufs != 0
        || !p.luts.is_empty()
        || p.sinks != [(p.ops.len() as u32, Len::Fixed(8))]
    {
        return false;
    }
    if p.ops.iter().enumerate().any(|(i, op)| op.a != i as u32) {
        return false;
    }
    let mut ops = p.ops.iter().peekable();
    let scan1 = if ops.peek().is_some_and(|i| i.op == Op::ScanAdd) {
        Some(ops.next().unwrap().imm)
    } else {
        None
    };
    let scan2 = if scan1.is_some() && ops.peek().is_some_and(|i| i.op == Op::ScanAdd) {
        Some(ops.next().unwrap().imm)
    } else {
        None
    };
    let shift = if ops.peek().is_some_and(|i| i.op == Op::ShlI) {
        ops.next().unwrap().imm as u32
    } else {
        0
    };
    let factor = if ops.peek().is_some_and(|i| i.op == Op::MulI) {
        ops.next().unwrap().imm
    } else {
        1
    };
    let add = if ops.peek().is_some_and(|i| i.op == Op::AddI) {
        ops.next().unwrap().imm
    } else {
        0
    };
    let float = ops.peek().is_some_and(|i| i.op == Op::I64ToF64);
    if float {
        ops.next();
    }
    let scale1 = if float
        && ops
            .peek()
            .is_some_and(|i| matches!(i.op, Op::F64DivI | Op::F64MulI))
    {
        ops.next().copied()
    } else {
        None
    };
    let scale2 = if scale1.is_some()
        && ops
            .peek()
            .is_some_and(|i| matches!(i.op, Op::F64DivI | Op::F64MulI))
    {
        ops.next().copied()
    } else {
        None
    };
    if ops.next().is_some() {
        return false;
    }
    #[inline(always)]
    fn run(buf: &mut [i64], scan1: Option<i64>, scan2: Option<i64>, f: impl Fn(i64) -> i64) {
        match (scan1, scan2) {
            (Some(mut a1), Some(mut a2)) => {
                for slot in buf {
                    a1 = a1.wrapping_add(*slot);
                    a2 = a2.wrapping_add(a1);
                    *slot = f(a2);
                }
            }
            (Some(mut a1), None) => {
                for slot in buf {
                    a1 = a1.wrapping_add(*slot);
                    *slot = f(a1);
                }
            }
            (None, _) => {
                for slot in buf {
                    *slot = f(*slot);
                }
            }
        }
    }
    let identity_int = shift == 0 && factor == 1 && add == 0;
    let integer = |v: i64| v.wrapping_shl(shift).wrapping_mul(factor).wrapping_add(add);
    match (float, scale1, scale2) {
        (false, _, _) if identity_int => {
            if scan1.is_some() {
                run(buf, scan1, scan2, |v| v);
            }
        }
        (false, _, _) => run(buf, scan1, scan2, integer),
        (true, None, _) if identity_int => {
            run(buf, scan1, scan2, |v| (v as f64).to_bits() as i64);
        }
        (true, None, _) => run(buf, scan1, scan2, |v| (integer(v) as f64).to_bits() as i64),
        (true, Some(s1), None) if s1.op == Op::F64DivI => {
            let d = f64::from_bits(s1.imm as u64);
            if identity_int {
                run(buf, scan1, scan2, |v| (v as f64 / d).to_bits() as i64);
            } else {
                run(buf, scan1, scan2, |v| {
                    (integer(v) as f64 / d).to_bits() as i64
                });
            }
        }
        (true, Some(s1), None) => {
            let m = f64::from_bits(s1.imm as u64);
            if identity_int {
                run(buf, scan1, scan2, |v| (v as f64 * m).to_bits() as i64);
            } else {
                run(buf, scan1, scan2, |v| {
                    (integer(v) as f64 * m).to_bits() as i64
                });
            }
        }
        (true, Some(s1), Some(s2)) => {
            let c1 = f64::from_bits(s1.imm as u64);
            let c2 = f64::from_bits(s2.imm as u64);
            let apply = |x: f64, op: Op, c: f64| if op == Op::F64DivI { x / c } else { x * c };
            if identity_int {
                run(buf, scan1, scan2, |v| {
                    apply(apply(v as f64, s1.op, c1), s2.op, c2).to_bits() as i64
                });
            } else {
                run(buf, scan1, scan2, |v| {
                    apply(apply(integer(v) as f64, s1.op, c1), s2.op, c2).to_bits() as i64
                });
            }
        }
    }
    true
}

/// Out-of-place wrapper retained for callers needing separate input/output slices.
pub(crate) fn eval_numeric_chain(p: &Program, input: &[i64], out: &mut [i64]) -> bool {
    if input.len() != out.len() {
        return false;
    }
    out.copy_from_slice(input);
    eval_numeric_chain_in_place(p, out)
}
