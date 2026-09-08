//! Offline generator for `src/resources/ice_noise_64.png`, the gray noise
//! texture the Ice material's triplanar bump-mapping samples (`iChannel0`).
//! Not part of the app's build or runtime - run with `cargo run --bin
//! generate_ice_noise` whenever the asset needs regenerating.
//!
//! Independent-per-texel gray noise: `ice_smooth_sample`
//! (`elemental_shader.wgsl`) does its own bicubic-style reconstruction on
//! top of unfiltered texel reads, so the smoothing is the shader's job, not
//! this texture's.
//!
//! Fixed-seeded so re-running it reproduces the same checked-in bytes.

const SIZE: u32 = 64;
const SEED: u64 = 0x1CE_5EED;

fn main() {
    let mut rng = fastrand::Rng::with_seed(SEED);
    let mut pixels = Vec::with_capacity((SIZE * SIZE * 4) as usize);
    for _ in 0..(SIZE * SIZE) {
        let value = rng.u8(..);
        pixels.extend_from_slice(&[value, value, value, 255]);
    }

    let path =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/resources/ice_noise_64.png");
    image::RgbaImage::from_raw(SIZE, SIZE, pixels)
        .expect("pixel buffer size mismatch")
        .save(&path)
        .unwrap_or_else(|err| panic!("failed to write {path:?}: {err}"));

    println!("wrote {path:?}");
}
