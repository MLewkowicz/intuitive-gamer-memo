from policies.policy_registry import instantiate_policy, POLICY_REGISTRY
from typing import List, Dict, Any
import pyspiel
from state_dataset.dataset import GameStateView, GameStateDataset



def load_game(game_config: Dict[str, Any]) -> pyspiel.Game:
    """Load and initialize a game from configuration."""
    game_name = game_config["name"]
    game_params = game_config.get("parameters", {})
    
    try:
        if game_params:
            game = pyspiel.load_game(game_name, game_params)
        else:
            game = pyspiel.load_game(game_name)
        print(f"✓ Loaded game: {game_name}")
        return game
    except Exception as e:
        raise RuntimeError(f"Failed to load game '{game_name}': {e}")


def load_policies(policies_config: List[Dict[str, Any]], game: pyspiel.Game, include_metadata=True) -> Dict[str, Any]:
    """Load and initialize policies from configuration."""
    policies = {}
    
    print(f"\nAvailable policies: {list(POLICY_REGISTRY.keys())}")
    print("Loading policies:")
    
    for i, policy_config in enumerate(policies_config):
        try:
            # Add the game parameter to policy config
            policy_config_with_game = policy_config.copy()
            if "parameters" not in policy_config_with_game:
                policy_config_with_game["parameters"] = {}
            policy_config_with_game["parameters"]["game"] = game
            
            policy = instantiate_policy(policy_config_with_game)
            policy_name = policy_config["name"]
            policy_id = f"{policy_name}_{i}" if policy_name in policies else policy_name
        
            if include_metadata:
                policies[policy_id] = {
                    "policy": policy,
                    "config": policy_config
                }
            else:
                policies[policy_id] = policy
            print(f"  ✓ {policy_id}: {policy.__class__.__name__}")
            
        except Exception as e:
            policy_name = policy_config.get("name", "unknown")
            print(f"  ✗ Failed to load policy '{policy_name}': {e}")
            continue
    
    if not policies:
        raise RuntimeError("No policies were successfully loaded")
    
    return policies


def build_sampler(cfg, game):
    ds = GameStateDataset(game)
    scfg = cfg.get("sampler",{})
    view = GameStateView(ds.all())
    
    print(f"Initial dataset size: {len(view)}")
        
    for p in scfg.get("predicates",[]):
        fn = p if callable(p) else eval(p)
        view = view.where(fn)
        print(f"After predicate: {len(view)} items")

    # read sampling parameters
    sample_cfg = scfg.get("sample", {})
    k = sample_cfg.get("k", 1)
    replace = sample_cfg.get("replace", True)
    
    print(f"Final dataset size: {len(view)}, sampling k={k}, replace={replace}")
    
    # Check if we have enough items for sampling without replacement
    if not replace and k > len(view):
        print(f"Warning: Cannot sample {k} items without replacement from {len(view)} items. Using replacement instead.")
        replace = True
    
    # Test sampling immediately to catch errors early
    print("Testing sample generation...")
    try:
        test_samples = view.sample(k=min(3, len(view)), replace=replace)
        print(f"Test sampling successful: got {len(test_samples)} samples")
        if test_samples:
            print(f"Sample structure: {list(test_samples[0].keys()) if test_samples[0] else 'Empty sample'}")
    except Exception as e:
        print(f"Error during test sampling: {e}")
        return None
    
    # attach sampling function to view    
    view.samples = lambda: view.sample(k=k, replace=replace)

    return view