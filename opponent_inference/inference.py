

from typing import Dict, Any
from utils.utils import load_policies
import numpy as np


def softmax(x):
    e = np.exp(x - np.max(x))   # subtract max for numerical stability
    return e / np.sum(e)

class OpponentInference:
    def __init__(self, candidate_policies, method="log_likelihood"):
        self.candidate_policies = candidate_policies
        self.fn = self.log_likelihoods if method == "log_likelihood" else self.agreement_counting
    
    def log_likelihoods(self, state_action_history) -> Dict[str, float]:
        # Implementation here
        log_likelihoods = {}
        for policy_name, policy in self.candidate_policies.items():
            log_likelihoods.setdefault(policy_name, 0.0)
            for state_action in state_action_history:
                state, action = state_action
                log_likelihoods[policy_name] += np.log(policy.action_likelihoods(state).get(action, 1e-10))
                # Compute log likelihoods based on policy
        
        policy_names = list(log_likelihoods.keys())
        log_vals = np.array(list(log_likelihoods.values()))

        policy_probs = dict(zip(policy_names, softmax(log_vals)))
        return policy_probs

    def agreement_counting(self, state_action_history) -> Dict[str, float]:
        # Implementation here
        return {}

    def calculate_likelihoods(self, state_action_history) -> Dict[str, float]:
        # Implementation here        
        return self.fn(state_action_history)
    

def load_inference(config: Dict[str, Any], game) -> OpponentInference:
    method = config.get("method", "log_likelihood")
    candidate_policy_cfgs = config.get("candidate_policies", [])
    candidate_policies = load_policies(candidate_policy_cfgs, game, include_metadata=False)



    # inference = OpponentInference(candidate_policies, method=method)
    # print(f"✓ Loaded Opponent Inference with method: {method}")
    # return inference
    return OpponentInference(candidate_policies, method=method)