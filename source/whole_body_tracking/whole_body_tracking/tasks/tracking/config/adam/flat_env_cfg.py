from isaaclab.utils import configclass

from whole_body_tracking.robots.adam import ADAM_ACTION_SCALE, ADAM_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg, GeneralTrackingEnvCfg


@configclass
class AdamFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.decimation = 10
        self.sim.dt = 0.002

        self.scene.robot = ADAM_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = ADAM_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso"
        self.commands.motion.body_names = [
            "pelvis",
            "hipRollLeft",
            "shinLeft",
            "toeLeft",
            "hipRollRight",
            "shinRight",
            "toeRight",
            "torso",
            "shoulderRollLeft",
            "elbowLeft",
            "shoulderRollRight",
            "elbowRight",
        ]


@configclass
class AdamFlatWoStateEstimationEnvCfg(AdamFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class AdamFlatLowFreqEnvCfg(AdamFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE


@configclass
class AdamFlatGeneralEnvCfg(GeneralTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.decimation = 10
        self.sim.dt = 0.002

        self.scene.robot = ADAM_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = ADAM_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso"
        self.commands.motion.body_names = [
            "pelvis",
            "hipRollLeft",
            "shinLeft",
            "toeLeft",
            "hipRollRight",
            "shinRight",
            "toeRight",
            "torso",
            "shoulderRollLeft",
            "elbowLeft",
            "shoulderRollRight",
            "elbowRight",
        ]


@configclass
class AdamFlatWoStateEstimationGeneralEnvCfg(AdamFlatGeneralEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class AdamFlatLowFreqGeneralEnvCfg(AdamFlatGeneralEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
