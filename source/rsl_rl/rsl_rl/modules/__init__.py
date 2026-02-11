# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions for neural-network components for RL-agents."""

from .actor_critic import ActorCritic
from .actor_critic_recurrent import ActorCriticRecurrent
from .normalizer import EmpiricalNormalization
from .rnd import RandomNetworkDistillation
from .student_teacher import StudentTeacher
from .student_teacher_recurrent import StudentTeacherRecurrent
from .actor_critic_transformer import ActorCriticTransformer
from .actor_critic_fsq import ActorCriticFSQ
from .actor_critic_vq import ActorCriticVQ
from .actor_critic_attention import ActorCriticAttention
from .residual_actor_critic import ResidualActorCritic
from .velocity_estimator import VelocityEstimator
from .velocity_estimator_transformer import VelocityEstimatorTransformer

__all__ = [
    "ActorCritic",
    "ActorCriticRecurrent",
    "EmpiricalNormalization",
    "RandomNetworkDistillation",
    "StudentTeacher",
    "StudentTeacherRecurrent",
    "ActorCriticTransformer",
    "ActorCriticFSQ",
    "ActorCriticVQ",
    "ActorCriticAttention",
    "ResidualActorCritic",
    "VelocityEstimator",
    "VelocityEstimatorTransformer",
]
