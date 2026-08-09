"""Reinforcement-learning package for the Gemma 4 reasoning study.

Modules that do not import torch (protocol, posterior, rewards, configs) are
unit-tested in CI. Torch-dependent modules (rollout, trainer, eval_policy) are
exercised on the training host.
"""
