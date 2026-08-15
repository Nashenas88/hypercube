//! Benches the CPU-only per-frame allocation sites named in
//! `perf_improvements.md` item #3, isolated from any wgpu/iced context via
//! `HypercubeShaderState::default()` (a solved cube, identity rotation, no
//! in-flight animation — the common steady-state case).

use criterion::{Criterion, criterion_group, criterion_main};
use hypercube::shader_widget::{
    HypercubeShaderProgram, HypercubeShaderState, sticker_instances_for_render,
};
use nalgebra::Matrix4;
use std::hint::black_box;

fn bench_calculate_indices(c: &mut Criterion) {
    let rotation_4d = Matrix4::identity();
    c.bench_function("calculate_indices", |b| {
        b.iter(|| HypercubeShaderProgram::calculate_indices(black_box(&rotation_4d)));
    });
}

fn bench_sticker_instances_for_render(c: &mut Criterion) {
    let state = HypercubeShaderState::default();
    c.bench_function("sticker_instances_for_render", |b| {
        b.iter(|| sticker_instances_for_render(black_box(&state)));
    });
}

criterion_group!(
    benches,
    bench_calculate_indices,
    bench_sticker_instances_for_render
);
criterion_main!(benches);
