from vlarlkit.policies.ppo import PPOPolicy
from vlarlkit.policies.dsrl import DSRLPolicy
from vlarlkit.policies.rlt import RLTPolicy

__all__ = ["PPOPolicy", "DSRLPolicy", "RLTPolicy"]


_OFFPOLICY_REGISTRY = {
    "dsrl": DSRLPolicy,
    "rlt": RLTPolicy,
}


def get_offpolicy_policy(cfg, model, target_model, rank):
    """Route to the correct off-policy policy class based on config."""
    policy_class = cfg.algorithm.get("policy_class", "dsrl")
    cls = _OFFPOLICY_REGISTRY[policy_class]
    return cls(cfg, model, target_model, rank)
