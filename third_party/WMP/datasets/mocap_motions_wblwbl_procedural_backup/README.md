# WBLWBL Dog V1 AMP Expert Motions

Converted from `/home/qyw/supermimic/amp_supermimic/data/training` for WMP `AMPLoader` (61-column JSON format, 50 Hz, `xyzw` root quaternion, body-frame `lin_vel` / `ang_vel`, joint order `RR, RL, FR, FL`).

Kept separate from A1's `datasets/mocap_motions/` (`FR, FL, RR, RL`).

## Clips

- `procedural_forward.txt` (MotionWeight 2.0): diagonal trot, `vx ≈ +0.55 m/s`, `z ≈ 0.391 m` (source: `forward/procedural_forward/procedural_forward.csv`)
- `procedural_turn_left.txt` (MotionWeight 1.0): diagonal trot turn left, `wz ≈ +0.85 rad/s` (source: `turn_left/procedural_turn_left/procedural_turn_left.csv`)
- `procedural_turn_right.txt` (MotionWeight 1.0): diagonal trot turn right, `wz ≈ -0.85 rad/s` (source: `turn_right/procedural_turn_right/procedural_turn_right.csv`)
