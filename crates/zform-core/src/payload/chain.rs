//! Linear numeric chains (`ProgramI64` with one input) lowered to one loop.

use super::ir::{Len, Op, Program, Width};

/// Lower a linear numeric SSA chain to one loop. The operations and constants still come
/// entirely from the payload; short columns avoid a handler transition for each arithmetic step.
pub(crate) fn eval_numeric_chain(p: &Program, input: &[i64], out: &mut [i64]) -> bool {
    if p.width != Width::W64
        || p.n_in != 1
        || p.n_bufs != 0
        || !p.luts.is_empty()
        || p.sinks != [(p.ops.len() as u32, Len::Fixed(8))]
        || input.len() != out.len()
    {
        return false;
    }
    if p.ops.iter().enumerate().any(|(i, op)| op.a != i as u32) {
        return false;
    }
    let mut ops = p.ops.iter().peekable();
    let scan = if ops.peek().is_some_and(|i| i.op == Op::ScanAdd) {
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
    let scale = if float
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
    fn run(input: &[i64], out: &mut [i64], scan: Option<i64>, f: impl Fn(i64) -> i64) {
        if let Some(mut accumulator) = scan {
            for (&v, output) in input.iter().zip(out) {
                accumulator = accumulator.wrapping_add(v);
                *output = f(accumulator);
            }
        } else {
            for (&v, output) in input.iter().zip(out) {
                *output = f(v);
            }
        }
    }
    let integer = |v: i64| v.wrapping_shl(shift).wrapping_mul(factor).wrapping_add(add);
    match (float, scale) {
        (false, _) => run(input, out, scan, integer),
        (true, None) => run(input, out, scan, |v| (integer(v) as f64).to_bits() as i64),
        (true, Some(i)) if i.op == Op::F64DivI => {
            let divisor = f64::from_bits(i.imm as u64);
            run(input, out, scan, |v| {
                (integer(v) as f64 / divisor).to_bits() as i64
            });
        }
        (true, Some(i)) => {
            let multiplier = f64::from_bits(i.imm as u64);
            run(input, out, scan, |v| {
                (integer(v) as f64 * multiplier).to_bits() as i64
            });
        }
    }
    true
}
