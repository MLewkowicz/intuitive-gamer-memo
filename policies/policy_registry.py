"""
Evaluator registry for managing different evaluation metrics.
"""

from typing import Dict, Any
from .intuitivegamer.policy import IntuitiveGamerPolicy
from .random.policy import RandomPolicy
from .mcts.policy import MCTSAgent

# Registry of available evaluators
POLICY_REGISTRY = {
    "intuitive_gamer": IntuitiveGamerPolicy,
    "random": RandomPolicy,
    "mcts": MCTSAgent,
}

def get_policy(name: str):
    """Get evaluator class by name."""
    if name not in POLICY_REGISTRY:
        raise KeyError(f"Unknown policy: {name}")
    return POLICY_REGISTRY[name]

def instantiate_policy(policy_config: Dict[str, Any]):
    """Instantiate a policy from configuration."""
    if isinstance(policy_config, str):
        policy_name = policy_config
        policy_params = {}
    else:
        policy_name = policy_config["name"]
        policy_params = policy_config.get("parameters", {})

    PolicyClass = get_policy(policy_name)
    return PolicyClass(**policy_params)