# camera.rs

3D orbit camera (`Camera`, `CameraController`, `Projection`). All three derive `Serialize`/`Deserialize` (plain data, no invariants broken by a round trip) so `snapshot.rs`'s `ViewSnapshot` can serialize a full view's camera state directly - `CameraController.distance` is the "zoom" it captures.

`CameraUniform` carries the combined `view_proj`, the translation-free `view_proj_inv` the skybox reprojects screen positions through, and `eye_position` - the camera's world-space position, `w` padded to 0. The surface materials derive their view direction as `normalize(-world_position)` instead, which places the eye at the world origin; that approximation is fine for a shading term but not for a material that traces a ray through its own volume, where it would flatten the parallax the ray depends on.
