#[path = "common/criterion_config.rs"]
mod criterion_config;

#[path = "common/fixtures.rs"]
mod fixtures;

#[path = "common/cpu.rs"]
mod cpu;

use cpu::{infer, preprocess, setup};
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn preprocess_bench(c: &mut Criterion) {
    let bench = setup();
    c.bench_function("preprocess", |b| {
        b.iter(|| black_box(preprocess(&bench).materialize().expect("materialize")));
    });
}

fn inference_bench(c: &mut Criterion) {
    let mut bench = setup();
    let input = preprocess(&bench)
        .materialize()
        .expect("materialize preprocess output");
    c.bench_function("inference", |b| {
        b.iter(|| black_box(infer(&mut bench, &input)));
    });
}

fn full_pass_bench(c: &mut Criterion) {
    let mut bench = setup();
    c.bench_function("full_pass", |b| {
        b.iter(|| {
            let pending = preprocess(&bench);
            black_box(infer(&mut bench, pending));
        });
    });
}

criterion_group! {
    name = cpu_pipeline;
    config = criterion_config::configured();
    targets = preprocess_bench, inference_bench, full_pass_bench
}
criterion_main!(cpu_pipeline);
