//! Shared animation-timing helpers.

/// Smoothstep ease: slow-fast-slow, applied to the normalized \[0,1\] progress.
pub(crate) fn ease(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Linear interpolation from `start` to `target` at normalized progress `t`.
pub(crate) fn lerp(start: f32, target: f32, t: f32) -> f32 {
    start + (target - start) * t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ease_maps_endpoints_and_midpoint() {
        assert_eq!(ease(0.0), 0.0);
        assert_eq!(ease(1.0), 1.0);
        assert_eq!(ease(0.5), 0.5);
    }

    #[test]
    fn ease_is_monotonic_increasing() {
        let samples: Vec<f32> = (0..=10).map(|i| i as f32 / 10.0).collect();
        for pair in samples.windows(2) {
            assert!(ease(pair[0]) <= ease(pair[1]));
        }
    }

    #[test]
    fn lerp_maps_endpoints_and_midpoint() {
        assert_eq!(lerp(2.0, 10.0, 0.0), 2.0);
        assert_eq!(lerp(2.0, 10.0, 1.0), 10.0);
        assert_eq!(lerp(2.0, 10.0, 0.5), 6.0);
    }
}
