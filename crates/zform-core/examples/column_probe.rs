//! Prints sampled physical columns and their template contexts for manual model investigation.
use std::{collections::HashSet, env, fs};
use zform_core::{
    framing::split_chunk,
    grammar::{PlanVariant, split_records_with_plan},
    plan::train_from_bytes,
};
fn main() {
    let args: Vec<_> = env::args().collect();
    let raw = fs::read(&args[1]).unwrap();
    let plan = train_from_bytes(&raw, &Default::default(), None);
    let fc = split_chunk(&raw, Default::default());
    let recs: Vec<_> = (0..fc.len()).map(|i| fc.record(i)).collect();
    let split = split_records_with_plan(&recs, &plan);
    for (i, c) in plan.columns.iter().enumerate() {
        if args.get(2).is_some_and(|needle| !c.name.contains(needle)) {
            continue;
        }
        let vals: Vec<_> = split.col_occs[i]
            .iter()
            .map(|v| v.as_slice(&split.scratch))
            .collect();
        println!(
            "COLUMN {i} {} rows={} distinct={} samples={:?}",
            c.name,
            vals.len(),
            vals.iter().collect::<HashSet<_>>().len(),
            vals.iter()
                .take(8)
                .map(|v| String::from_utf8_lossy(v))
                .collect::<Vec<_>>()
        );
        for (j, v) in plan.variants.iter().enumerate() {
            if let PlanVariant::Shape(s) = v {
                if s.slots.contains(&(i as u16)) {
                    let mut text = String::new();
                    for (k, &slot) in s.slots.iter().enumerate() {
                        text += &String::from_utf8_lossy(&s.literals[k]);
                        text += &format!("<{}>", plan.columns[slot as usize].name);
                    }
                    text += &String::from_utf8_lossy(s.literals.last().unwrap());
                    println!("  {j}: {text}");
                }
            }
        }
    }
}
