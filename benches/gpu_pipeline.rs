#[path = "common/fixtures.rs"]
mod fixtures;

#[path = "common/gpu.rs"]
mod gpu;

use criterion::{criterion_group, criterion_main, Criterion};
use gpu::{infer, preprocess, setup};
use std::hint::black_box;

fn preprocess_bench(c: &mut Criterion) {
    let bench = setup();
    c.bench_function("preprocess", |b| {
        b.iter(|| black_box(preprocess(&bench)));
    });
}

fn inference_bench(c: &mut Criterion) {
    let mut bench = setup();
    let input = preprocess(&bench);
    c.bench_function("inference", |b| {
        b.iter(|| black_box(infer(&mut bench, &input)));
    });
}

fn full_pass_bench(c: &mut Criterion) {
    let mut bench = setup();
    c.bench_function("full_pass", |b| {
        b.iter(|| {
            let input = preprocess(&bench);
            black_box(infer(&mut bench, &input));
        });
    });
}

criterion_group!(gpu_pipeline, preprocess_bench, inference_bench, full_pass_bench);
criterion_main!(gpu_pipeline);
