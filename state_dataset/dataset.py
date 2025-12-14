import numpy as np
import pyspiel
import random
from tqdm import tqdm  # Import tqdm
import pdb
# Default directions if game doesn't specify them
DEFAULT_DIRECTIONS = [(1,0), (0,1), (1,1), (1,-1)] # h, v, diag, anti

def extract_board(state, game):
    """
    Extracts board from OpenSpiel observation tensor.
    Assumes shape (3, rows, cols) with planes: [Player(X), Opponent(O), Empty].
    Returns board where: 1=CurrentPlayer, -1=Opponent, 0=Empty.
    """
    shape = game.observation_tensor_shape()
    obs = np.array(state.observation_tensor(state.current_player())).reshape(shape)
    
    # Plane 0: Current Player
    # Plane 1: Opponent 
    me_plane = obs[0, :, :]
    opp_plane = obs[1, :, :]    
    return me_plane - opp_plane

def longest_chain(state, player_val, game):
    """
    Return max contiguous stones for `player` on board.
    player_val: 1 for current player, -1 for opponent (relative to state.current_player)
    """
    best = 0
    board = extract_board(state, game)
    rows, cols = board.shape
    
    # 1. Determine Player ID to fetch valid directions
    current_p_id = state.current_player()
    
    if player_val == 1:
        target_p_id = current_p_id
    else:
        target_p_id = 1 - current_p_id
        
    # 2. Get Valid Directions from Game (if supported)
    if hasattr(game, "get_valid_directions"):
        directions = game.get_valid_directions(target_p_id)
    else:
        directions = DEFAULT_DIRECTIONS

    # 3. Calculate Chain Length
    for r in range(rows):
        for c in range(cols):
            if board[r, c] != player_val: 
                continue
            for dr, dc in directions:
                length = 1
                rr, cc = r + dr, c + dc
                while 0 <= rr < rows and 0 <= cc < cols and board[rr, cc] == player_val:
                    length += 1
                    rr += dr; cc += dc
                best = max(best, length)
    return best

def generate_all_states(ds, state, visited, num_turns, game, max_depth=None, pbar=None):
    """
    Recursively visits states. 
    Added max_depth optional parameter to prevent infinite recursion on larger boards.
    Added pbar to track progress.
    """
    key = str(state) 
    if key in visited or state.is_terminal():
        return
        
    if max_depth is not None and num_turns > max_depth:
        return
    
    # Calculate chains
    len_chain_me = longest_chain(state, 1, game)
    len_chain_opp = longest_chain(state, -1, game)

    info = {
        'state': state.clone(),
        'longest_chain_me': len_chain_me,
        'longest_chain_opp': len_chain_opp,
        'freespace': len(state.legal_actions()),
        'winning': len_chain_me > len_chain_opp, 
        'tied': len_chain_me == len_chain_opp,
        'losing': len_chain_me < len_chain_opp,
        'current_player': state.current_player(),
        'num_turns': num_turns 
    }
    
    visited[key] = info
    ds.add(info)
    
    # Update progress bar
    if pbar is not None:
        pbar.update(1)

    for a in state.legal_actions():
        child = state.clone()
        child.apply_action(a)
        generate_all_states(ds, child, visited, num_turns + 1, game, max_depth, pbar)


class GameStateDataset:
    def __init__(self, game: pyspiel.Game, max_depth_limit=None):
        self._items = []    # list of dicts
        self.game = game
        self.max_depth_limit = max_depth_limit
        
        # Start generation
        print(f"Generating states for {game}...")
        visited = {}
        
        # Initialize tqdm context manager
        # Note: total is unknown for recursion, so it will show iterations/sec
        with tqdm(desc="Generating States", unit=" states") as pbar:
            generate_all_states(self, game.new_initial_state(), visited, 0, game, max_depth=max_depth_limit, pbar=pbar)
            
        print(f"Generated {len(self._items)} unique non-terminal states.")

    def add(self, info):
        """info is the dict you are currently putting in visited[key]"""
        self._items.append(info)

    # --- Query builder ----------------------------------------------------
    def filter(self, **kwargs):
        out = []
        for x in self._items:
            try:
                if all(x.get(k) == v for k, v in kwargs.items()):
                    out.append(x)
            except Exception as e:
                continue
        return GameStateView(out) 

    def where(self, fn):
        return GameStateView([x for x in self._items if fn(x)])

    def all(self):
        return list(self._items)


class GameStateView:
    def __init__(self, items):
        self._items = items

    def filter(self, **kwargs):
        out = []
        for x in self._items:
            try:
                if all(x.get(k) == v for k, v in kwargs.items()):
                    out.append(x)
            except Exception:
                continue
        return GameStateView(out)

    def where(self, fn):
        return GameStateView([x for x in self._items if fn(x)])

    def sample(self, k=1, replace=True):
        if not self._items:
            print("Warning: Trying to sample from empty dataset")
            return []
        
        if not replace and k > len(self._items):
            print(f"Warning: Requesting {k} samples from {len(self._items)} items without replacement. Returning all items.")
            return random.sample(self._items, len(self._items))
            
        try:
            if replace:
                return random.choices(self._items, k=k)   # with replacement
            else:
                return random.sample(self._items, k=k)    # without replacement
        except Exception as e:
            print(f"Error during sampling: {e}")
            return []

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        yield from self._items

if __name__ == "__main__":
    try:
        from games.mnk_game import MNKGame
        # Example: 3x3, K=3, No Diagonals
        rules = {"allowed_directions": {0: ["h", "v"], 1: ["h", "v"]}}
        game = MNKGame(m=3, n=3, k=3, rules=rules)
        print("Loaded Custom MNK Game (No Diagonals)")
    except ImportError:
        print("Custom MNKGame not found, loading standard Tic-Tac-Toe")
        game = pyspiel.load_game("tic_tac_toe")

    # Generate dataset (limited depth to keep test fast)
    dataset = GameStateDataset(game, max_depth_limit=9)
    
    # Test filtering
    winning_states = dataset.filter(winning=True)
    print(f"Winning states found: {len(winning_states)}")
    
    samples = winning_states.sample(k=3)
    for s in samples:
        print("\nSampled State:")
        print(s['state'])
        print(f"Me Chain: {s['longest_chain_me']}, Opp Chain: {s['longest_chain_opp']}")