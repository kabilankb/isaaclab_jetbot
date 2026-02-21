# IsaacLabTutorial - JetBot Navigation RL Environment

A reinforcement learning environment built on **NVIDIA Isaac Sim** and **Isaac Lab** where a JetBot robot learns to navigate towards randomly-commanded directions using PPO/AMP algorithms.

Based on the official [Isaac Lab Tutorial](https://isaac-sim.github.io/IsaacLab).

## Prerequisites

- **NVIDIA Isaac Sim 4.5.0+** ([installation guide](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/index.html))
- **Isaac Lab** framework ([installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html))
- **NVIDIA GPU** with CUDA support
- **Python 3.10+**

## Quick Start

### 1. Install the package

```bash
./launch.sh install
# or
make install
```

### 2. Verify the environment is registered

```bash
./launch.sh list-envs
```

### 3. Test with a random agent

```bash
./launch.sh random-agent
# or with custom options
./launch.sh random-agent --num_envs 64 --device cuda:0
```

### 4. Train an RL agent

```bash
# Train with PPO (default)
./launch.sh train --algorithm PPO --num_envs 100

# Train with AMP
./launch.sh train --algorithm AMP --num_envs 50

# Resume from checkpoint
./launch.sh train --algorithm PPO --checkpoint logs/skrl/.../checkpoints/best_agent.pt
```

### 5. Evaluate a trained agent

```bash
# Auto-detect latest checkpoint
./launch.sh play --algorithm PPO

# Specify checkpoint
./launch.sh play --checkpoint path/to/checkpoint.pt

# Real-time evaluation
./launch.sh play --algorithm PPO --real-time
```

## Launch Commands Reference

| Command | Description |
|---------|-------------|
| `./launch.sh install` | Install package in editable mode |
| `./launch.sh list-envs` | List all registered environments |
| `./launch.sh random-agent` | Run with random actions |
| `./launch.sh zero-agent` | Run with zero actions (baseline) |
| `./launch.sh train` | Train RL agent with skrl |
| `./launch.sh play` | Evaluate trained checkpoint |
| `./launch.sh check` | Verify prerequisites |
| `./launch.sh help` | Show full usage details |

You can also use `make`:

```bash
make help                           # Show all targets
make train ALGORITHM=PPO NUM_ENVS=100
make play CHECKPOINT=path/to/model.pt
make train-ppo                      # Shortcut for PPO training
make train-amp                      # Shortcut for AMP training
```

## Environment Details

| Property | Value |
|----------|-------|
| **Environment ID** | `Template-Isaac-Lab-Tutorial-Direct-v0` |
| **Robot** | NVIDIA JetBot (2-wheel differential drive) |
| **Action Space** | 2D (left/right wheel velocities) |
| **Observation Space** | 3D (alignment dot product, cross product, forward speed) |
| **Reward** | forward_speed + alignment_with_command |
| **Episode Length** | 5 seconds |
| **Simulation Rate** | 120 Hz |
| **Default Parallel Envs** | 100 |

## Project Structure

```
IsaacLabTutorial/
├── launch.sh                      # Main launch script
├── Makefile                       # Make-based shortcuts
├── scripts/
│   ├── list_envs.py               # List registered environments
│   ├── random_agent.py            # Random action baseline
│   ├── zero_agent.py              # Zero action baseline
│   └── skrl/
│       ├── train.py               # RL training (PPO/AMP)
│       └── play.py                # Checkpoint evaluation
└── source/
    └── isaac_lab_tutorial/
        ├── setup.py               # Package configuration
        └── isaac_lab_tutorial/
            ├── robots/
            │   └── jetbot.py      # JetBot robot config
            └── tasks/
                └── direct/
                    └── isaac_lab_tutorial/
                        ├── isaac_lab_tutorial_env.py      # Environment implementation
                        ├── isaac_lab_tutorial_env_cfg.py   # Environment config
                        └── agents/
                            ├── skrl_ppo_cfg.yaml          # PPO hyperparameters
                            └── skrl_amp_cfg.yaml          # AMP hyperparameters
```

## Training Logs

Training logs, checkpoints, and videos are saved to `logs/skrl/`. Structure:

```
logs/skrl/<experiment>/
├── <timestamp>_ppo_torch/
│   ├── params/              # Saved configs (env.yaml, agent.yaml)
│   ├── checkpoints/         # Model checkpoints
│   └── videos/              # Recorded videos (if --video flag used)
```

## Supported Algorithms

| Algorithm | Description | Config |
|-----------|-------------|--------|
| **PPO** | Proximal Policy Optimization | `skrl_ppo_cfg.yaml` |
| **AMP** | Adversarial Motion Priors | `skrl_amp_cfg.yaml` |
| **IPPO** | Independent PPO (multi-agent) | via skrl |
| **MAPPO** | Multi-Agent PPO | via skrl |

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
