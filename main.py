import yaml
import argparse
import pyspiel
import os
from typing import List, Dict, Any

# Imports for custom components
from policies.policy_registry import instantiate_policy, POLICY_REGISTRY
from state_dataset.dataset import GameStateView, GameStateDataset

# Import your custom game class
from games.mnk_game import MNKGame 

def load_game(game_config: Dict[str, Any]) -> pyspiel.Game:
    """Load and initialize a game from configuration."""
    game_name = game_config["name"]
    game_params = game_config.get("parameters", {})
    
    try:
        # --- FIX: Check for custom python games first ---
        if game_name == "mnk_game":
            print(f"✓ Loading Custom Python Game: {game_name}")
            # Unpack parameters (m, n, k, rules) into the constructor
            return MNKGame(**game_params)
        # ------------------------------------------------
        
        # Fallback to standard OpenSpiel C++ games
        if game_params:
            game = pyspiel.load_game(game_name, game_params)
        else:
            game = pyspiel.load_game(game_name)
            
        print(f"✓ Loaded OpenSpiel game: {game_name}")
        return game
        
    except Exception as e:
        raise RuntimeError(f"Failed to load game '{game_name}': {e}")


def load_policies(policies_config: List[Dict[str, Any]], game: pyspiel.Game) -> Dict[str, Any]:
    """Load and initialize policies from configuration."""
    policies = {}
    
    print(f"\nAvailable policies: {list(POLICY_REGISTRY.keys())}")
    print("Loading policies:")
    
    for i, policy_config in enumerate(policies_config):
        try:
            # Create a copy to inject the game instance
            policy_config_with_game = policy_config.copy()
            if "parameters" not in policy_config_with_game:
                policy_config_with_game["parameters"] = {}
            policy_config_with_game["parameters"]["game"] = game
            
            policy = instantiate_policy(policy_config_with_game)
            policy_name = policy_config["name"]
            
            # Handle duplicate names (e.g. random vs random)
            policy_id = f"{policy_name}_{i}" if policy_name in policies else policy_name
        
            # Store the wrapper dict that analysis.py expects
            policies[policy_id] = {
                "policy": policy,
                "config": policy_config
            }
            print(f"  ✓ {policy_id}: {policy.__class__.__name__}")
            
        except Exception as e:
            policy_name = policy_config.get("name", "unknown")
            print(f"  ✗ Failed to load policy '{policy_name}': {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not policies:
        raise RuntimeError("No policies were successfully loaded")
    
    return policies

def build_sampler(cfg, game):
    # Pass the game instance to the dataset generator
    ds = GameStateDataset(game)
    scfg = cfg.get("sampler",{})
    
    view = GameStateView(ds.all())
    
    print(f"Initial dataset size: {len(view)}")
        
    for p in scfg.get("predicates",[]):
        try:
            fn = p if callable(p) else eval(p)
            view = view.where(fn)
        except Exception as e:
            print(f"Warning: Failed to apply predicate '{p}': {e}")

    print(f"After predicates: {len(view)} items")

    # read sampling parameters
    sample_cfg = scfg.get("sample", {})
    k = sample_cfg.get("k", 1)
    replace = sample_cfg.get("replace", True)
    
    print(f"Sampling k={k}, replace={replace}")
    
    # Safety check for sampling
    if not replace and k > len(view):
        print(f"Warning: Cannot sample {k} items without replacement from {len(view)} items. Using replacement instead.")
        replace = True
    
    # Attach sampling function
    view.samples = lambda: view.sample(k=k, replace=replace)

    return view

def run_experiment(config: Dict[str, Any]) -> None:
    """Run the full experiment based on configuration."""
    print(f"Running experiment: {config['experiment_name']}")
    print(f"Description: {config['description']}")
    
    # 1. Load Game
    print(f"\n{'='*50}")
    print("GAME SETUP")
    print('='*50)
    game = load_game(config["game"])
    
    # 2. Load Policies
    print(f"\n{'='*50}")
    print("POLICY SETUP")
    print('='*50)
    policies = load_policies(config["policies"], game)

    # 3. Build Sampler
    print("\nBuilding sampler...")
    sampler = build_sampler(config, game)
    
    if sampler is None or len(sampler) == 0:
        print("Error: Sampler yielded 0 states. Check predicates or game generation.")
        return
    
    print("Importing analysis modules...")
    from analysis import Experiment, visualize_agreement_heatmap
    
    print("Creating experiment...")
    
    try:
        # FIX: Pass 'policies' directly (the dict of dicts)
        # analysis.py expects to access policy['policy'], so we must not strip the wrapper.
        exp = Experiment(policies, sampler)
        exp_one = exp.run_max_action_disagreement()
        
        # Run Pairwise Comparison
        exp = Experiment(policies, sampler)
        exp_two = exp.run_pairwise_comparison()
        
        if exp_one:
            print("Visualizing results...")
            visualize_agreement_heatmap([exp_one, exp_two], titles=["Max Action Agreement", "Pairwise Action Agreement"])
        else:
            print("No results to visualize")
            
    except Exception as e:
        print(f"Error during experiment: {e}")
        import traceback
        traceback.print_exc()
    

def main():
    parser = argparse.ArgumentParser(description="Run game policy comparison experiments")
    parser.add_argument("--config", "-c", type=str, required=True, 
                        help="Path to experiment configuration file")

    args = parser.parse_args()
    
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        run_experiment(config)
        
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config}' not found")
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in configuration file: {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()