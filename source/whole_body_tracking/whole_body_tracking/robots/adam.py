import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

ARMATURE_PND_50_14_50 = 0.15075
ARMATURE_PND_30_14_50 = 0.04075
ARMATURE_PND_20_14_50 = 0.016075
ARMATURE_PND_60_17_50 = 0.225
ARMATURE_PND_130_92_7 = 0.13426
ARMATURE_PND_80_20_30 = 0.2637
ARMATURE_PND_50_52_30 = 0.0549

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_PND_50_14_50 = ARMATURE_PND_50_14_50 * NATURAL_FREQ**2
STIFFNESS_PND_30_14_50 = ARMATURE_PND_30_14_50 * NATURAL_FREQ**2
STIFFNESS_PND_20_14_50 = ARMATURE_PND_20_14_50 * NATURAL_FREQ**2
STIFFNESS_PND_60_17_50 = ARMATURE_PND_60_17_50 * NATURAL_FREQ**2
STIFFNESS_PND_130_92_7 = ARMATURE_PND_130_92_7 * NATURAL_FREQ**2
STIFFNESS_PND_80_20_30 = ARMATURE_PND_80_20_30 * NATURAL_FREQ**2
STIFFNESS_PND_50_52_30 = ARMATURE_PND_50_52_30 * NATURAL_FREQ**2

DAMPING_PND_50_14_50 = 2.0 * DAMPING_RATIO * ARMATURE_PND_50_14_50 * NATURAL_FREQ
DAMPING_PND_30_14_50 = 2.0 * DAMPING_RATIO * ARMATURE_PND_30_14_50 * NATURAL_FREQ
DAMPING_PND_20_14_50 = 2.0 * DAMPING_RATIO * ARMATURE_PND_20_14_50 * NATURAL_FREQ
DAMPING_PND_60_17_50 = 2.0 * DAMPING_RATIO * ARMATURE_PND_60_17_50 * NATURAL_FREQ
DAMPING_PND_130_92_7 = 2.0 * DAMPING_RATIO * ARMATURE_PND_130_92_7 * NATURAL_FREQ
DAMPING_PND_80_20_30 = 2.0 * DAMPING_RATIO * ARMATURE_PND_80_20_30 * NATURAL_FREQ
DAMPING_PND_50_52_30 = 2.0 * DAMPING_RATIO * ARMATURE_PND_50_52_30 * NATURAL_FREQ

ADAM_CYLINDER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/pnd_description/urdf/adam/adam_lite_agx.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.83),               
        joint_pos={
            # legs joints
            'hipPitch_Left': -0.586,
            'hipRoll_Left': -0.085,
            'hipYaw_Left': -0.322,
            'kneePitch_Left': 1.288,
            'anklePitch_Left': -0.789,
            'ankleRoll_Left': 0.002,

            'hipPitch_Right': -0.586,
            'hipRoll_Right': 0.085,
            'hipYaw_Right': 0.322,
            'kneePitch_Right': 1.288,
            'anklePitch_Right': -0.789,
            'ankleRoll_Right': -0.002,

            'waistRoll': 0.0,
            'waistPitch': 0.0,
            'waistYaw': 0.0,

            # arms joints
            'shoulderPitch_Left':0.0,
            'shoulderRoll_Left':0.1,
            'shoulderYaw_Left':0.0,
            'elbow_Left':-0.3,

            'shoulderPitch_Right':0.0,
            'shoulderRoll_Right':-0.1,
            'shoulderYaw_Right':0.0,
            'elbow_Right':-0.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "hipPitch_.*",
                "hipRoll_.*",
                "hipYaw_.*",
                "kneePitch_.*",
            ],
            effort_limit_sim={
                "hipPitch_.*": 230.0,
                "hipRoll_.*": 160.0,
                "hipYaw_.*": 105.0,
                "kneePitch_.*": 230.0,
            },
            velocity_limit_sim={
                "hipPitch_.*": 15.0,
                "hipRoll_.*": 8.0,
                "hipYaw_.*": 8.0,
                "kneePitch_.*": 15.0,
            },
            stiffness={
                "hipPitch_.*": STIFFNESS_PND_130_92_7,
                "hipRoll_.*": STIFFNESS_PND_80_20_30,
                "hipYaw_.*": STIFFNESS_PND_60_17_50,
                "kneePitch_.*": STIFFNESS_PND_130_92_7,
            },
            damping={
                "hipPitch_.*": DAMPING_PND_130_92_7,
                "hipRoll_.*": DAMPING_PND_80_20_30,
                "hipYaw_.*": DAMPING_PND_60_17_50,
                "kneePitch_.*": DAMPING_PND_130_92_7,
            },
            armature={
                "hipPitch_.*": ARMATURE_PND_130_92_7,
                "hipRoll_.*": ARMATURE_PND_80_20_30,
                "hipYaw_.*": ARMATURE_PND_60_17_50,
                "kneePitch_.*": ARMATURE_PND_130_92_7,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=40.0,
            velocity_limit_sim=20.0,
            joint_names_expr=["anklePitch_.*", "ankleRoll_.*"],
            stiffness=2.0 * STIFFNESS_PND_50_52_30,
            damping=2.0 * DAMPING_PND_50_52_30,
            armature=2.0 * ARMATURE_PND_50_52_30,
        ),
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=110.0,
            velocity_limit_sim=8.0,
            joint_names_expr=["waistRoll", "waistPitch", "waistYaw"],
            stiffness=2.0 * STIFFNESS_PND_60_17_50,
            damping=2.0 * DAMPING_PND_60_17_50,
            armature=2.0 * ARMATURE_PND_60_17_50,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulderPitch_.*",
                "shoulderRoll_.*",
                "shoulderYaw_.*",
                "elbow_.*",
            ],
            effort_limit_sim={
                "shoulderPitch_.*": 65.0,
                "shoulderRoll_.*": 65.0,
                "shoulderYaw_.*": 65.0,
                "elbow_.*": 30.0,
            },
            velocity_limit_sim={
                "shoulderPitch_.*": 8.0,
                "shoulderRoll_.*": 8.0,
                "shoulderYaw_.*": 8.0,
                "elbow_.*": 8.0,
            },
            stiffness={
                "shoulderPitch_.*": STIFFNESS_PND_50_14_50,
                "shoulderRoll_.*": STIFFNESS_PND_50_14_50,
                "shoulderYaw_.*": STIFFNESS_PND_30_14_50,
                "elbow_.*": STIFFNESS_PND_30_14_50,
            },
            damping={
                "shoulderPitch_.*": DAMPING_PND_50_14_50,
                "shoulderRoll_.*": DAMPING_PND_50_14_50,
                "shoulderYaw_.*": DAMPING_PND_30_14_50,
                "elbow_.*": DAMPING_PND_30_14_50,
            },
            armature={
                "shoulderPitch_.*": ARMATURE_PND_50_14_50,
                "shoulderRoll_.*": ARMATURE_PND_50_14_50,
                "shoulderYaw_.*": ARMATURE_PND_30_14_50,
                "elbow_.*": ARMATURE_PND_30_14_50,
            },
        ),
    },
)

ADAM_ACTION_SCALE = {}
for a in ADAM_CYLINDER_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            ADAM_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
