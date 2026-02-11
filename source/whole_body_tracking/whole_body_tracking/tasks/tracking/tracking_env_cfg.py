from __future__ import annotations

from dataclasses import MISSING
from typing import Union

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg

##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.tracking.mdp as mdp

##
# Scene definition
##

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

from isaaclab.terrains import TerrainGeneratorCfg, MeshPlaneTerrainCfg, HfRandomUniformTerrainCfg
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="plane",
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #     ),
    #     visual_material=sim_utils.MdlFileCfg(
    #         mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
    #         project_uvw=True,
    #     ),
    # )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator", 
        terrain_generator=TerrainGeneratorCfg(
            seed=42,
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=10,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            sub_terrains={
                "flat": MeshPlaneTerrainCfg(proportion=0.5),
                "slightly_rough": HfRandomUniformTerrainCfg(
                    proportion=0.5,
                    noise_range=(0.01, 0.03),
                    noise_step=0.01,
                ),
            },
        ),
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    # robots
    robot: ArticulationCfg = MISSING
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0, debug_vis=True
    )


##
# MDP settings
##


@configclass
class SingleMotionCommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
    )

@configclass
class MultiMotionCommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MultiMotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        resample_motions_every_s = 1e9,
        motion_sampling_warmup_s=1e9,
        motion_sampling_ramp_s=1e9,
        motion_sampling_schedule="cosine",
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.25, n_max=0.25)
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        ref_base_lin_vel = ObsTerm(func=mdp.ref_base_lin_vel_b, params={"command_name": "motion"})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        # def __post_init__(self):
        #     self.history_length = 5

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()

@configclass
class ObservationsExpertCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        ref_base_lin_vel = ObsTerm(func=mdp.ref_base_lin_vel_b, params={"command_name": "motion"})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)


        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            # self.history_length = 5

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        ref_base_lin_vel = ObsTerm(func=mdp.ref_base_lin_vel_b, params={"command_name": "motion"})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        # def __post_init__(self):
        #     self.history_length = 5

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)

    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )

@configclass
class RewardsExpertCfg:
    """
    Expert reward configuration - MOSAIC.
    """

    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.5,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.5,
        params={"command_name": "motion", "std": 3.14},
    )
    motion_anchor_lin_vel = RewTerm(
        func=mdp.motion_anchor_linear_velocity_error_exp,
        weight=1.0,  # 2*1.0
        params={"command_name": "motion", "std": 1.0},
    )

    teleop_body_position_extend = RewTerm(
        func=mdp.teleop_body_position_extend,
        weight=1.0,
        params={
            "command_name": "motion",
            "upper_body_std": 0.5, 
            "lower_body_std": 0.5,  
            "upper_weight": 1.0,
            "lower_weight": 1.0,
        }
    )
    teleop_vr_3point = RewTerm(
        func=mdp.teleop_vr_3point,
        weight=0.5,
        params={"command_name": "motion", "std": 0.5}  
    )
    teleop_body_position_feet = RewTerm(
        func=mdp.teleop_body_position_feet,
        weight=1,  # 1.5*1
        params={"command_name": "motion", "std": 0.5} 
    )
    teleop_body_rotation_extend = RewTerm(
        func=mdp.teleop_body_rotation_extend,
        weight=0.5,
        params={"command_name": "motion", "std": 0.5} 
    )
    teleop_body_ang_velocity_extend = RewTerm(
        func=mdp.teleop_body_ang_velocity_extend,
        weight=0.5,
        params={"command_name": "motion", "std": 3.14}
    )
    teleop_body_velocity_extend = RewTerm(
        func=mdp.teleop_body_velocity_extend,
        weight=0.5,
        params={"command_name": "motion", "std": 1.0} 
    )

    # ===== Penalty terms (same as base) =====
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)  # 2*1e-1
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)  # -2*2.5e-7
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)  # -2*1e-5
    
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    motion_end = DoneTerm(func=mdp.motion_end, params={"command_name": "motion"}, time_out=True)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


##
# Environment configuration
##


@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=8192, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: SingleMotionCommandsCfg = SingleMotionCommandsCfg()
    
    # MDP settings
    rewards: RewardsExpertCfg = RewardsExpertCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 15 * 2**17
        # viewer settings
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.viewer.origin_type = "env"
        self.viewer.asset_name = "robot"
    

@configclass
class GeneralTrackingEnvCfg(TrackingEnvCfg):
    """Configuration for the general tracking environment."""

    commands: MultiMotionCommandsCfg = MultiMotionCommandsCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        self.commands = MultiMotionCommandsCfg()
        
@configclass
class ExpertGeneralTrackingEnvCfg(GeneralTrackingEnvCfg):
    """
    Expert general tracking environment.
    """

    commands: MultiMotionCommandsCfg = MultiMotionCommandsCfg()
    observations: ObservationsExpertCfg = ObservationsExpertCfg()
    rewards: RewardsExpertCfg = RewardsExpertCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        self.commands = MultiMotionCommandsCfg()

@configclass
class DistillationTrackingEnvCfg(GeneralTrackingEnvCfg):
    """
    Student-teacher distillation environment configuration.
    """

    commands: MultiMotionCommandsCfg = MultiMotionCommandsCfg()

    @configclass
    class DistillationObservationsCfg:
        """Observation specifications for distillation."""

        @configclass
        class PolicyCfg(ObsGroup):
            """Observations for policy group."""
            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
            motion_anchor_ori_b = ObsTerm(
                func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
            )
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
            joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
            joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
            actions = ObsTerm(func=mdp.last_action)

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True
                self.history_length = 5

        @configclass
        class TeacherCfg(ObsGroup):
            """Teacher observations - teacehr information."""
            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
            motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
            motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
            body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
            body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
            base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
            ref_base_lin_vel = ObsTerm(func=mdp.ref_base_lin_vel_b, params={"command_name": "motion"})
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
            joint_pos = ObsTerm(func=mdp.joint_pos_rel)
            joint_vel = ObsTerm(func=mdp.joint_vel_rel)
            actions = ObsTerm(func=mdp.last_action)

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        @configclass
        class PrivilegedCfg(ObsGroup):
            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
            motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
            motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
            body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
            body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
            base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
            ref_base_lin_vel = ObsTerm(func=mdp.ref_base_lin_vel_b, params={"command_name": "motion"})
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
            joint_pos = ObsTerm(func=mdp.joint_pos_rel)
            joint_vel = ObsTerm(func=mdp.joint_vel_rel)
            actions = ObsTerm(func=mdp.last_action)

            # def __post_init__(self):
            #     self.history_length = 5

        @configclass
        class RefVelEstimatorCfg(ObsGroup):
            """Observations for reference velocity estimator."""
            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})  # [58] = [joint_pos(29), joint_vel(29)]
            ref_projected_gravity = ObsTerm(func=mdp.ref_projected_gravity, params={"command_name": "motion"})  # [3] using ANCHOR BODY quaternion

            def __post_init__(self):
                self.enable_corruption = False 
                self.concatenate_terms = True
                self.history_length = 5

        # observation groups
        policy: PolicyCfg = PolicyCfg()
        teacher: TeacherCfg = TeacherCfg()
        critic: PrivilegedCfg = PrivilegedCfg()
        ref_vel_estimator: RefVelEstimatorCfg = RefVelEstimatorCfg() 

    observations: DistillationObservationsCfg = DistillationObservationsCfg()
    rewards: RewardsExpertCfg = RewardsExpertCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        self.commands = MultiMotionCommandsCfg()



@configclass
class OneStageTrackingEnvCfg(GeneralTrackingEnvCfg):
    """
    Teacher-student distillation environment configuration.
    """

    commands: MultiMotionCommandsCfg = MultiMotionCommandsCfg()

    @configclass
    class OneStageObservationsCfg:
        """Observation specifications for distillation."""

        @configclass
        class PolicyCfg(ObsGroup):
            """Observations for policy group."""
            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
            motion_anchor_ori_b = ObsTerm(
                func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
            )
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
            joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
            joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
            actions = ObsTerm(func=mdp.last_action)

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True
                self.history_length = 5

        @configclass
        class PrivilegedCfg(ObsGroup):
            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
            motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
            motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
            body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
            body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
            base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
            ref_base_lin_vel = ObsTerm(func=mdp.ref_base_lin_vel_b, params={"command_name": "motion"})
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
            joint_pos = ObsTerm(func=mdp.joint_pos_rel)
            joint_vel = ObsTerm(func=mdp.joint_vel_rel)
            actions = ObsTerm(func=mdp.last_action)

            # def __post_init__(self):
            #     self.history_length = 5
            

        # observation groups
        policy: PolicyCfg = PolicyCfg()
        critic: PrivilegedCfg = PrivilegedCfg()

    observations: OneStageObservationsCfg = OneStageObservationsCfg()
    rewards: RewardsExpertCfg = RewardsExpertCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        self.commands = MultiMotionCommandsCfg()



@configclass
class MultiDistillationTrackingEnvCfg(GeneralTrackingEnvCfg):
    """
    Teacher-student distillation environment configuration.
    """

    commands: MultiMotionCommandsCfg = MultiMotionCommandsCfg()

    @configclass
    class DistillationObservationsCfg:
        """Observation specifications for distillation."""

        @configclass
        class PolicyCfg(ObsGroup):
            """Observations for policy group."""
            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
            motion_anchor_ori_b = ObsTerm(
                func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
            )
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
            joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
            joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
            actions = ObsTerm(func=mdp.last_action)

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True
                self.history_length = 5

        @configclass
        class TeacherCfg(ObsGroup):
            """Observations for policy group."""
            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
            motion_anchor_ori_b = ObsTerm(
                func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
            )
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
            joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
            joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
            actions = ObsTerm(func=mdp.last_action)

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True
                self.history_length = 5


                
        @configclass
        class PrivilegedCfg(ObsGroup):
            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
            motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
            motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
            body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
            body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
            base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
            ref_base_lin_vel = ObsTerm(func=mdp.ref_base_lin_vel_b, params={"command_name": "motion"})
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
            joint_pos = ObsTerm(func=mdp.joint_pos_rel)
            joint_vel = ObsTerm(func=mdp.joint_vel_rel)
            actions = ObsTerm(func=mdp.last_action)


        @configclass
        class RefVelEstimatorCfg(ObsGroup):
            """Observations for reference velocity estimator.
            """
            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})  # [58] = [joint_pos(29), joint_vel(29)]
            ref_projected_gravity = ObsTerm(func=mdp.ref_projected_gravity, params={"command_name": "motion"})  # [3] using ANCHOR BODY quaternion

            def __post_init__(self):
                self.enable_corruption = False 
                self.concatenate_terms = True
                self.history_length = 5

        # observation groups
        policy: PolicyCfg = PolicyCfg()
        teacher: TeacherCfg = TeacherCfg()
        critic: PrivilegedCfg = PrivilegedCfg()
        ref_vel_estimator: RefVelEstimatorCfg = RefVelEstimatorCfg()

    observations: DistillationObservationsCfg = DistillationObservationsCfg()
    rewards: RewardsExpertCfg = RewardsExpertCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        self.commands = MultiMotionCommandsCfg()

