from policies.base import GamePolicy
import numpy as np
import pyspiel
from typing import Dict

class IntuitiveGamerPolicy(GamePolicy):

    def _uaux(self, state: pyspiel.State, action: int) -> float:
        """Compute auxiliary utility for a given action based on auxiliary utility defined in intuitive gamer paper."""
        return 1.0
    
    def _uself(self, state: pyspiel.State, action: int) -> float:
        """Compute the self utility for a given action to ensure that the agent wins as quickly as possible."""
        return 1.0

    def _uopp(self, state: pyspiel.State, action: int) -> float:
        """Compute the opponent utility for a given action to ensure that the opponent does not win."""
        return 1.0

    def action_likelihoods(self, state: pyspiel.State) -> Dict[int, float]:
        likelihoods = {}
        for action in state.legal_actions():
            likelihoods[action] = 1.0

        return likelihoods