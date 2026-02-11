# Expert Trajectory Collection Module

This module provides tools for collecting expert trajectories with flexible student observation configurations for MOSAIC's off-policy BC learning.

## 📁 Module Structure

```
whole_body_tracking/collection/
├── __init__.py                        # Module exports
├── expert_trajectory_collector.py     # Trajectory collector
└── student_obs_builder.py            # Student observation builder
```

## 🔧 Components

### ExpertTrajectoryCollector

Collects and saves expert trajectories in a format compatible with MOSAIC.

**Features**:
- Supports multiple observation types (teacher, student)
- Stores action distributions (mean, sigma)
- Efficient numpy-based storage
- Compatible with MOSAIC's expert BC

**Usage**:
```python
from whole_body_tracking.collection import ExpertTrajectoryCollector

collector = ExpertTrajectoryCollector(
    obs_dim=154,
    action_dim=29,
    student_obs_dim=770,  # Optional
    max_trajectories=1000,
    device='cuda:0'
)

# During collection
collector.add_step(
    observations=teacher_obs,
    actions=actions,
    student_observations=student_obs,  # Optional
    valid_mask=active_mask  # Optional mask to mark active envs
)

# Mark trajectory ends
collector.mark_trajectory_end(count=num_done)

# Save
collector.save('expert_data/expert.npy')
```

### StudentObsBuilder

Builds student observations with isolated history buffers.

**Features**:
- Isolated history management (doesn't modify environment)
- Flexible observation composition
- Automatic history updates and resets
- Support for arbitrary history lengths

**Usage**:
```python
from whole_body_tracking.collection import StudentObsBuilder

builder = StudentObsBuilder(
    env=env,
    obs_keys=['command', 'joint_pos', 'joint_vel', 'actions'],
    history_length=5,
    device='cuda:0'
)

# During collection
student_obs = builder.get_observation(obs_dict)
builder.update_history(obs_dict)

# On episode reset
builder.reset_history(done_env_ids)
```

## 📊 Data Format

### Saved Trajectory File (.npy)

```python
{
    'observations': [T, N, obs_dim],           # Teacher observations
    'student_observations': [T, N, student_obs_dim],  # Student observations (optional)
    'actions': [T, N, action_dim],
    'action_mean': [T, N, action_dim],
    'action_sigma': [T, N, action_dim],
    'valid_mask': [T, N],  # Optional, True for active envs
    'metadata': {
        'num_trajectories': int,
        'num_steps': int,
        'obs_dim_dict': dict,
        'action_dim': int,
        'has_valid_mask': bool,
    }
}
```

## 🚀 Quick Start

See `docs/EXPERT_COLLECTION.md` for detailed usage instructions.

**Basic collection**:
```bash
python scripts/rsl_rl/collect_expert_trajectories.py \
    --task=Expert-General-Tracking-Flat-G1-v0 \
    --checkpoint_path=logs/rsl_rl/g1_flat/model_40000.pt \
    --motion=/path/to/motions \
    --output_path=expert_data/expert_history5.npy \
    --num_envs=1000 \
    --max_steps=500000 \
    --student_obs_history=5 \
    --headless
```

## 🔗 Integration with MOSAIC

After collecting trajectories, use them in MOSAIC training:

```python
# rsl_rl_mosaic_cfg.py
algorithm = RslRlMOSAICAlgorithmCfg(
    hybrid=True,
    use_ppo=True,
    expert_trajectory_path="expert_data/expert_history5.npy",
    lambda_off_policy=0.3,
    off_policy_batch_size=256,
)
```

## 📝 Notes

- **History Length**: Must match your student policy's configuration
- **Observation Keys**: Automatically extracted from environment's policy group
- **Memory Usage**: Scales with history length and number of steps
- **Collection Time**: ~10-20 minutes for 500k steps with 1000 envs

## 🔍 See Also

- `docs/EXPERT_COLLECTION.md` - Detailed documentation
- `scripts/collect_expert.sh` - Example collection scenarios
- `scripts/rsl_rl/collect_expert_trajectories.py` - Main collection script
