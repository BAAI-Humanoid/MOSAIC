"""
Expert Trajectory Collection Module

This module provides tools for collecting expert trajectories with flexible
student observation configurations for MOSAIC's off-policy BC learning.

Key Components:
- ExpertTrajectoryCollector: Collects and saves expert trajectories
- StudentObsBuilder: Builds student observations with history support
"""

from .expert_trajectory_collector import ExpertTrajectoryCollector
from .student_obs_builder import StudentObsBuilder, create_student_obs_builder_from_env_cfg

__all__ = [
    "ExpertTrajectoryCollector",
    "StudentObsBuilder",
    "create_student_obs_builder_from_env_cfg",
]
