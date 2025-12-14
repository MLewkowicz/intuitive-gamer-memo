from typing import Dict, Any
import pyspiel
from .mnk_game import MNKGame

def load_custom_game(game_config: Dict[str, Any]) -> Any:
    """
    Loads a game from config.
    Supports custom "mnk_game" or standard OpenSpiel games.
    """
    game_name = game_config["name"]
    game_params = game_config.get("parameters", {})

    if game_name == "mnk_game":
        # Unpack parameters (m, n, k, rules)
        return MNKGame(**game_params)
    
    # Fallback to standard OpenSpiel
    try:
        if game_params:
            return pyspiel.load_game(game_name, game_params)
        else:
            return pyspiel.load_game(game_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load game '{game_name}': {e}")