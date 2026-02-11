import os

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
from whole_body_tracking.tasks.tracking.mdp.commands import MultiMotionCommand


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"
    ):
        super().__init__(env, train_cfg, log_dir, device)
        # Note: Teacher policy is automatically loaded in MOSAIC.__init__ if teacher_checkpoint_path is set

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"

            # Get velocity estimator info if available
            ref_vel_estimator = None
            ref_vel_estimator_obs_dim = None

            if hasattr(self.alg, 'ref_vel_estimator') and self.alg.ref_vel_estimator is not None:
                ref_vel_estimator = self.alg.ref_vel_estimator
                if hasattr(self.alg, 'ref_vel_estimator_obs_shape') and self.alg.ref_vel_estimator_obs_shape is not None:
                    ref_vel_estimator_obs_dim = self.alg.ref_vel_estimator_obs_shape[0]

            # if the command is a multi motion command, or a single motion command
            export_motion_policy_as_onnx(
                self.alg.policy,
                normalizer=self.obs_normalizer,
                path=policy_path,
                filename=filename,
                ref_vel_estimator=ref_vel_estimator,
                ref_vel_estimator_obs_dim=ref_vel_estimator_obs_dim,
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
