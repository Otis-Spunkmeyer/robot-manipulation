# Robot Manipulation Build

13-week build toward robot-learning and robotics-hardware roles: a real SO-101 arm under ROS 2, a simulation twin, learned policies with measured success rates, and a designed, built and tested electromechanical subsystem.

## Stack
Ubuntu 24.04 · ROS 2 Jazzy · Gazebo Harmonic · MuJoCo · ManiSkill3 · PyTorch · LeRobot · RTX 4070 (12 GB)

## Hardware
SO-101 leader/follower (12x Feetech STS3215; 12 V follower)

## Progress
| Week | Focus | Result |
|---|---|---|
| 1 | Setup, MuJoCo, ManiSkill3 | Panda in MuJoCo at 308x real time; ManiSkill3 GPU sim ~35k env-steps/s at n=256 |

## Repo layout
- `labs/`: weekly lab scripts
- `ros2_ws/src/`: ROS 2 packages
- `envs/`: conda environment files (`conda env create -f envs/sim.yml`)
- `docs/`: weekly write-ups
