from vlarlkit.policies.ppo import PPOPolicy
from vlarlkit.policies.grpo import GRPOPolicy
from vlarlkit.policies.dsrl import DSRLPolicy
from vlarlkit.policies.rlt import RLTPolicy
from vlarlkit.policies.qrflow import QRFlowPolicy

__all__ = [
    "PPOPolicy",
    "GRPOPolicy",
    "DSRLPolicy",
    "RLTPolicy",
    "QRFlowPolicy",
    "get_onpolicy_policy",
    "get_offpolicy_policy",
]


_ONPOLICY_REGISTRY = {
    "ppo": PPOPolicy,
    "grpo": GRPOPolicy,
}


_OFFPOLICY_REGISTRY = {
    "dsrl": DSRLPolicy,
    "rlt": RLTPolicy,
}


def get_onpolicy_policy(cfg, model, rank):
    """Route to the correct on-policy policy class based on config."""
    policy_class = cfg.algorithm.get("policy_class", "ppo")
    cls = _ONPOLICY_REGISTRY[policy_class]
    return cls(cfg, model, rank)


def get_offpolicy_policy(cfg, model, target_model, rank):
    """Route to the correct off-policy policy class based on config."""
    policy_class = cfg.algorithm.get("policy_class", "dsrl")
    cls = _OFFPOLICY_REGISTRY[policy_class]
    return cls(cfg, model, target_model, rank)
