# ray_casting.rs

CPU-side ray/AABB/triangle intersection against 4D→3D-projected stickers, for hover and click picking. `ray_sticker_intersection` and `calculate_sticker_aabb` are `pub(crate)` beyond that: `shader_widget::FireSticker::is_behind` reuses both, casting rays through a Fire/Ice/Light sticker's own corners to test occlusion against another such sticker's real geometry rather than an approximating scalar key.
