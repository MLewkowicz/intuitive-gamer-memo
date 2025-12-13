import yaml
import argparse
from typing import List, Dict, Any
from utils.utils import load_game, load_policies, build_sampler


def run_experiment(config: Dict[str, Any]) -> None:
    """Run the full experiment based on configuration."""
    print(f"Running experiment: {config['experiment_name']}")
    print(f"Description: {config['description']}")
    
    # Load game
    print(f"\n{'='*50}")
    print("GAME SETUP")
    print('='*50)
    game = load_game(config["game"])
    
    # Load policies  
    print(f"\n{'='*50}")
    print("POLICY SETUP")
    print('='*50)
    policies = load_policies(config["policies"], game)

    print("\nBuilding sampler...")
    sampler = build_sampler(config, game)
    
    if sampler is None:
        print("Error: Failed to build sampler")
        return
    
    print("Importing analysis modules...")
    from analysis import Experiment, visualize_agreement_heatmap
    
    print("Creating experiment...")
    experiment = Experiment(policies, sampler)

    print("Running pairwise comparison...")
    try:
        experiment = Experiment(policies, sampler)
        exp_one = experiment.run_max_action_disagreement()
        experiment = Experiment(policies, sampler)
        exp_two = experiment.run_pairwise_comparison()
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


if __name__ == "__main__":
    main()
