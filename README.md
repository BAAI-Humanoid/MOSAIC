# MOSAIC Training Code

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://opensource.org/license/apache-2-0)

[[Website]](https://baai-humanoid.github.io/MOSAIC/)
[[Arxiv]](https://arxiv.org/pdf/2602.08594)
[[Dataset]](https://huggingface.co/datasets/BAAI-Humanoid/MOSAIC_Dataset)

## Overview

MOSAIC is an open-source system for generalist humanoid motion tracking and whole-body teleoperation across multiple interfaces.
It trains a teleoperation-oriented tracker with rewards that emphasize global motion consistency for stable, long-horizon behaviors.
To bridge sim-to-real interface gaps, it uses rapid residual adaptation to inject interface-specific corrections while preserving generality.

This repo focuses on the teleoperation policy training; for sim-to-sim and sim-to-real deployment, see [RobotBridge](https://github.com/BAAI-Humanoid/RobotBridge).

## Installation

- Install Isaac Lab v2.1.0 by following
  the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend
  using the conda installation as it simplifies calling Python scripts from the terminal.

  ```bash
  # create virtual environment
  conda create -n isaaclab python=3.10 -y
  conda activate isaaclab
  pip install --upgrade pip

  # install PyTorch
  pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

  # install isaacsim 4.5
  pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com

  # verify the isaacsim installation
  isaacsim 

  ## isaaclab
  # download
  git clone https://github.com/isaac-sim/IsaacLab.git
  cd IsaacLab

  # switch to version 2.1.0
  git fetch --all
  git checkout v2.1.0 

  # install
  ./isaaclab.sh --install

  # verify
  ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

  ```

- Clone this repository separately from the Isaac Lab installation (i.e., outside the `IsaacLab` directory):

```bash
# Option 1: SSH (main branch only)
git clone -b main --single-branch git@github.com:BAAI-Humanoid/MOSAIC.git

# Option 2: HTTPS (main branch only)
git clone -b main --single-branch https://github.com/BAAI-Humanoid/MOSAIC.git
```

- Using a Python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e source/whole_body_tracking
```
and install the algorithm library

```bash
python -m pip install -e source/rsl_rl
```

## General Training Pipeline

### Motion Preprocessing

We store all processed motions locally. The reference motion should be retargeted and use generalized
coordinates only.

- Gather the reference motion datasets (please follow the original licenses), we use the same convention as .csv of
  [Unitree's dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)

- Convert retargeted motions to include the maximum coordinates information (body pose, body velocity, and body
  acceleration) via forward kinematics:
  
  - Single Motion

  ```bash
  python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
  ```

  The processed motion file will be saved locally with output name {motion_name}.

  - Multiple Motions
  ```bash
  python scripts/batch_csv_to_npz.py --input_dir {motion_path} --input_fps 30 --output_dir {motion_save_path} --headless
  ```
  This converts the processed motion files to the output target path {motion_save_path}.

- Replay motions from local files in Isaac Sim:

```bash
python scripts/replay_npz.py --motion_file={motion_file_path}
```

### Policy Training

- Train policy by the following command for single motion:

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--motion the path to the motion file \
--headless --logger wandb --log_project_name {project_name} --run_name {run_name}
```
- Train policy by the following command for multiple motion:

```bash
python scripts/rsl_rl/train.py --task=General-Tracking-Flat-G1-v0 \
--motion the path to the motion folder \
--headless --logger wandb --log_project_name {project_name} --run_name {run_name}
```

- If you want to activate multiple GPUs for training in parallel, just follow the command like that:
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/rsl_rl/train.py --task=General-Tracking-Flat-G1-v0 --distributed \
--headless --logger wandb --log_project_name {project_name} --run_name {run_name} \
--motion the path to the motion folder
```

### Policy Evaluation

- Play the trained policy by the following command for single motion:

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
```

- Play the trained policy by the following command for multiple motion:

```bash
python scripts/rsl_rl/play.py --task=General-Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
```

The WandB run path can be located in the run overview. It follows the format {your_organization}/{project_name}/ along
with a unique 8-character identifier. Note that run_name is different from run_path.

## MOSAIC Training

This section follows the full training flow in order. Before training, prepare the following motion sources:
1) Public datasets downloaded from their official sites: [AMASS](https://amass.is.tue.mpg.de/) and [OMOMO](https://github.com/lijiaman/omomo_release).
2) Built-in MOSAIC datasets (optical MoCap, inertial MoCap, and [GENMO](https://github.com/NVlabs/GENMO)-generated motions) downloaded from [our open dataset](https://huggingface.co/datasets/BAAI-Humanoid/MOSAIC_Dataset).
3) Adaptor training data (VR and inertial MoCap) downloaded from [our open dataset](https://huggingface.co/datasets/BAAI-Humanoid/MOSAIC_Dataset).

Training is provided as shell scripts. Run them in order, and make sure to set the motion source paths inside each script before running.

1) MOSAIC GMT policy training using multi-source dataset

```bash
bash run/run_mosaic_gmt.sh
```
After training, you get the GMT model: `gmt_checkpoint.pt`

1) MOSAIC teleoperation adaptor training using adaptor training data

```bash
bash run/run_mosaic_adaptor.sh
```

After training, you get the adaptor model: `adaptor_checkpoint.pt`

Then update the motion path and model path for GMT and Adaptor in with relative task config:
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py`
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_mosaic_cfg.py`

3) MOSAIC multi-teacher residual distillation (fast adaptation)

```bash
bash run/run_mosaic_residual_adaptation.sh
```

## Code Structure

Below is an overview of the code structure for this repository:

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**
  This directory contains the atomic functions to define the MDP for MOSAIC. Below is a breakdown of the functions:

    - **`commands.py`**
      Command library to compute relevant variables from the reference motion, current robot state, and error
      computations. This includes pose and velocity error calculation, initial state randomization, and adaptive
      sampling.

    - **`rewards.py`**
      Implements the DeepMimic reward functions and smoothing terms.

    - **`events.py`**
      Implements domain randomization terms.

    - **`observations.py`**
      Implements observation terms for motion tracking and data collection.

    - **`terminations.py`**
      Implements early terminations and timeouts.

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**
  Contains the environment (MDP) hyperparameters configuration for the tracking task.

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**
  Contains the PPO hyperparameters for the tracking task.

- **`source/whole_body_tracking/whole_body_tracking/robots`**
  Contains robot-specific settings, including armature parameters, joint stiffness/damping calculation, and action scale
  calculation.

- **`scripts`**
  Includes utility scripts for preprocessing motion data, training policies, and evaluating trained policies.

This structure is designed to ensure modularity and ease of navigation for developers expanding the project.

## Supported Task List

Currently, three robot platform `G1/H1_2/Adam` are supported. The G1 registry in `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/__init__.py` exposes the following Isaac Lab task IDs.

| Task ID | Description |
| --- | --- |
| `Tracking-Flat-G1-v0` | Single-motion tracking baseline using `G1FlatEnvCfg` with the standard control frequency and state-estimation inputs. |
| `Tracking-Flat-G1-Wo-State-Estimation-v0` | Single-motion variant without state-estimation inputs via `G1FlatWoStateEstimationEnvCfg`. |
| `Tracking-Flat-G1-Low-Freq-v0` | Single-motion setup that lowers the control frequency (`G1FlatLowFreqEnvCfg` + `G1FlatLowFreqPPORunnerCfg`). |
| `General-Tracking-Flat-G1-v0` | Multi-motion training environment configured by `G1FlatGeneralEnvCfg`. |
| `General-Tracking-Flat-G1-Wo-State-Estimation-v0` | Multi-motion variant without state-estimation inputs, based on `G1FlatWoStateEstimationGeneralEnvCfg`. |
| `General-Tracking-Flat-G1-Low-Freq-v0` | Multi-motion environment with reduced control frequency using `G1FlatLowFreqGeneralEnvCfg` and `G1FlatLowFreqPPORunnerCfg`. |
| `Expert-General-Tracking-Flat-G1-v0` | Expert multi-motion environment using `G1FlatExpertGeneralEnvCfg` with the standard PPO runner. |
| `Expert-General-Tracking-Flat-G1-MOSAIC-v0` | Expert multi-motion environment using `G1FlatExpertGeneralEnvCfg` with `G1FlatMOSAICRunnerCfg`. |
| `Distillation-General-Tracking-Flat-G1-v0` | Distillation environment using `G1DistillationTrackingEnvCfg` with `G1FlatDistillationRunnerCfg`. |
| `MOSAIC-Distill-General-Tracking-Flat-G1-v0` | Distillation environment using `G1DistillationTrackingEnvCfg` with `G1FlatMOSAICHybridRunnerCfg`. |
| `MOSAIC-Pure-Distill-General-Tracking-Flat-G1-v0` | Distillation environment using `G1DistillationTrackingEnvCfg` with `G1FlatMOSAICPureDistillationRunnerCfg`. |
| `MOSAIC-RL-Continue-General-Tracking-Flat-G1-v0` | Distillation environment using `G1DistillationTrackingEnvCfg` with `G1FlatMOSAICRLContinueRunnerCfg`. |
| `MOSAIC-Residual-General-Tracking-Flat-G1-v0` | Distillation environment using `G1DistillationTrackingEnvCfg` with `G1FlatMOSAICRLContinueResidualRunnerCfg`. |
| `MOSAIC-MultiTeacher-Residual-Tracking-Flat-G1-v0` | Multi-teacher distillation environment using `G1MultiDistillationTrackingEnvCfg` with `G1FlatMOSAICMultiTeacherResidualRunnerCfg`. |
| `General-Tracking-Flat-G1-Wo-State-Estimation-v0-World-Coordinate-Reward` | One-stage tracking variant using `G1OneStageTrackingEnvCfg` with the standard PPO runner. |

To introduce additional tasks, add new `gym.register` entries in the same file and point them to the corresponding `EnvCfg` and PPO runner configurations.

## Acknowledgements

This project is built on the following open-source projects:

- [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
