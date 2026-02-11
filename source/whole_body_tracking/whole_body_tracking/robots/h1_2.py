import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

ARMATURE_N7520_14p3 = 0.01017752
ARMATURE_N7520_22p5 = 0.025101925
ARMATURE_N5020_16 = 0.003609725
ARMATURE_M107_15 = 0.063259741
ARMATURE_M107_24 = 0.160478022

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_N7520_14p3 = ARMATURE_N7520_14p3 * (NATURAL_FREQ/2)**2
STIFFNESS_N7520_22p5 = ARMATURE_N7520_22p5 * NATURAL_FREQ**2
STIFFNESS_N5020_16 = ARMATURE_N5020_16 * (NATURAL_FREQ/10)**2
STIFFNESS_M107_15 = ARMATURE_M107_15 * NATURAL_FREQ**2
STIFFNESS_M107_24 = ARMATURE_M107_24 * NATURAL_FREQ**2

DAMPING_N7520_14p3 = 2.0 * DAMPING_RATIO * ARMATURE_N7520_14p3 * NATURAL_FREQ
DAMPING_N7520_22p5 = 2.0 * DAMPING_RATIO * ARMATURE_N7520_22p5 * NATURAL_FREQ
DAMPING_N5020_16 = 2.0 * DAMPING_RATIO * ARMATURE_N5020_16 * NATURAL_FREQ
DAMPING_M107_15 = 2.0 * DAMPING_RATIO * ARMATURE_M107_15 * NATURAL_FREQ
DAMPING_M107_24 = 2.0 * DAMPING_RATIO * ARMATURE_M107_24 * NATURAL_FREQ

H1_2_CYLINDER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/unitree_description/urdf/h1_2/h1_2_handless.urdf",
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
        pos=(0.0, 0.0, 1.05),               
        joint_pos={
            # legs joints
            "left_hip_yaw_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_pitch_joint": -0.16, 
            "left_knee_joint": 0.36,    
            "left_ankle_pitch_joint": -0.15,
            "left_ankle_roll_joint": 0.0,
            
            "right_hip_yaw_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.16,
            "right_knee_joint": 0.36,
            "right_ankle_pitch_joint": -0.15,
            "right_ankle_roll_joint": 0.0,
            
            "torso_joint": 0.0,
            # arms joints
            "left_shoulder_pitch_joint": 0.4, 
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.3,          
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            
            "right_shoulder_pitch_joint": 0.4,  
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.3, 
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 200.0,
                ".*_hip_roll_joint": 300.0,
                ".*_hip_pitch_joint": 300.0,
                ".*_knee_joint": 300.0,
            },
            velocity_limit_sim=100.0,
            stiffness={
                ".*_hip_yaw_joint": STIFFNESS_M107_15,
                ".*_hip_roll_joint": STIFFNESS_M107_24,
                ".*_hip_pitch_joint": STIFFNESS_M107_24,
                ".*_knee_joint": STIFFNESS_M107_24,
            },
            damping={
                ".*_hip_yaw_joint": DAMPING_M107_15,
                ".*_hip_roll_joint": DAMPING_M107_24,
                ".*_hip_pitch_joint": DAMPING_M107_24,
                ".*_knee_joint": DAMPING_M107_24,
            },
            armature={
                ".*_hip_yaw_joint": ARMATURE_M107_15,
                ".*_hip_roll_joint": ARMATURE_M107_24,
                ".*_hip_pitch_joint": ARMATURE_M107_24,
                ".*_knee_joint": ARMATURE_M107_24,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=75.0,
            velocity_limit_sim=100.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=2.0 * STIFFNESS_N7520_14p3,
            damping=2.0 * DAMPING_N7520_14p3,
            armature=2.0 * ARMATURE_N7520_14p3,
        ),
        "torso": ImplicitActuatorCfg(
            effort_limit_sim=200.0,
            velocity_limit_sim=100.0,
            joint_names_expr=["torso_joint"],
            stiffness=STIFFNESS_M107_15,
            damping=DAMPING_M107_15,
            armature=ARMATURE_M107_15,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 100.0,
                ".*_shoulder_roll_joint": 100.0,
                ".*_shoulder_yaw_joint": 100.0,
                ".*_elbow_joint": 100.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 25.0,
                ".*_wrist_yaw_joint": 25.0,
            },
            velocity_limit_sim=100.0,
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_N7520_22p5,
                ".*_shoulder_roll_joint": STIFFNESS_N7520_22p5,
                ".*_shoulder_yaw_joint": STIFFNESS_N7520_22p5,
                ".*_elbow_joint": STIFFNESS_N7520_22p5,
                ".*_wrist_roll_joint": STIFFNESS_N5020_16,
                ".*_wrist_pitch_joint": STIFFNESS_N5020_16,
                ".*_wrist_yaw_joint": STIFFNESS_N5020_16,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_N7520_22p5,
                ".*_shoulder_roll_joint": DAMPING_N7520_22p5,
                ".*_shoulder_yaw_joint": DAMPING_N7520_22p5,
                ".*_elbow_joint": DAMPING_N7520_22p5,
                ".*_wrist_roll_joint": DAMPING_N5020_16,
                ".*_wrist_pitch_joint": 2*DAMPING_N5020_16,
                ".*_wrist_yaw_joint": 2*DAMPING_N5020_16,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_N7520_22p5,
                ".*_shoulder_roll_joint": ARMATURE_N7520_22p5,
                ".*_shoulder_yaw_joint": ARMATURE_N7520_22p5,
                ".*_elbow_joint": ARMATURE_N7520_22p5,
                ".*_wrist_roll_joint": ARMATURE_N5020_16,
                ".*_wrist_pitch_joint": 2*ARMATURE_N5020_16,
                ".*_wrist_yaw_joint": 2*ARMATURE_N5020_16,
            },
        ),
    },
)

H1_2_ACTION_SCALE = {}
for a in H1_2_CYLINDER_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            H1_2_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
