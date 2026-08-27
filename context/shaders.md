# shaders/*.wgsl

WGSL shaders sharing `Transform4D` (`rotation_matrix`, `viewer_distance`, `sticker_scale`, `face_gap`, `face_gap_4d`, `elapsed_seconds`), `CameraUniform`, and `StickerInstance` structs plus 4D math functions, all defined once in `math4d.wgsl` and pulled into each pipeline shader via `naga_oil`'s `#import` (composed in `renderer.rs` through a `naga_oil::compose::Composer`, since WGSL itself has no import mechanism). `sticker_common.wgsl` similarly shares `LightUniform` and `HighlightingUniform` (and their bound globals) across every sticker material shader.
