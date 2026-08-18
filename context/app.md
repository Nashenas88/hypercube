# app.rs

`HypercubeApp` holds only UI-control state (scale/gap sliders, render mode, settings, the reveal toggle's runtime state). Builds a left control panel plus a right `Shader::new(HypercubeShaderProgram)` viewport. Contains no 3D/4D logic. The sticker-scale/face-gap sliders are hidden behind a "Reveal"/"Hide" toggle button: pressing it bumps a `reveal_generation` counter (same one-shot generation-counter pattern `reset_generation` uses to reach `HypercubeShaderState`) and flips `revealed` immediately, while `reveal_animating` disables the button and hides the sliders until `shader_widget.rs` publishes `Message::RevealAnimationComplete` back once the flourish settles.

The "1/2/3 Random Moves" and "Scramble" buttons all send `Message::RandomMoves(count)`, which bumps `random_moves_generation` and stores `count` in `pending_random_move_count` - a payload carried alongside the counter the same way `revealed` accompanies `reveal_generation`, since a bare generation bump can't carry data on its own.

`Message::Reset` bumps `reset_generation` and sets `reset_animating`, which - mirroring `reveal_animating` - disables the Reset/Random Move(s)/Scramble buttons until `shader_widget.rs` publishes `Message::ResetAnimationComplete` once the 4D-orientation-to-identity animation settles.
