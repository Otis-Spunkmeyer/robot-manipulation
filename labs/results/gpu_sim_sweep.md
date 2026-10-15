# ManiSkill3 GPU sim sweep (Week 1)
PickCube-v1, physx_cuda, obs=state, pd_joint_delta_pos, RTX 4070 12 GB, 32 GB RAM. Metric: env.step+env.reset, 1000 parallel steps.

| n envs | env-steps/s | per-env steps/s | GPU mem (GB) | CPU RAM (GB) |
|---|---|---|---|---|
| 256 | 34,971 | 136.6 | 1.40 | 3.0 |
| 1024 | 114,554 | 111.9 | 1.60 | 4.6 |
| 2048 | 180,198 | 88.0 | 1.86 | 6.8 |
| 4096 | 196,600 | 48.0 | 2.30 | 11.1 |

Knee at n = 2048: 92% of peak throughput at 61% of the n = 4096 host RAM. GPU compute saturates first; VRAM is not limiting; host RAM grows fastest. Week 8 default: 2048 envs (state obs); re-measure with RGB obs.
