import numpy as np
import pyspiel

class MNKGameState:
    def __init__(self, game, board_shape, rules):
        # REMOVED: super().__init__(game) -> caused the crash
        self._game = game
        self._rows, self._cols = board_shape
        self._rules = rules
        
        # Board: -1=empty, 0=Player1 (X), 1=Player2 (O)
        self._board = np.full(board_shape, -1, dtype=int)
        
        self._current_player = 0
        self._game_over = False
        self._returns = [0.0, 0.0]
        self._move_count = 0
        
        # Track opening moves (e.g. "Player 2 moves twice on first turn")
        self._moves_remaining_in_turn = self._get_opening_moves(0)

    def _get_opening_moves(self, player_id):
        if self._move_count <= 1:
            opening_moves = self._rules.get("opening_moves", {})
            return opening_moves.get(player_id, 1)
        return 1

    def current_player(self):
        return self._current_player if not self._game_over else pyspiel.PlayerId.TERMINAL

    def legal_actions(self):
        if self._game_over: return []
        return [i for i, val in enumerate(self._board.flatten()) if val == -1]

    def apply_action(self, action):
        if self._game_over:
            raise RuntimeError("Cannot apply action to terminal state")

        row = action // self._cols
        col = action % self._cols
        
        self._board[row, col] = self._current_player
        self._move_count += 1
        self._moves_remaining_in_turn -= 1

        outcome = self._check_outcome(self._current_player)
        
        if outcome is not None:
            self._game_over = True
            if outcome == 1: # Win
                self._returns[self._current_player] = 1.0
                self._returns[1 - self._current_player] = -1.0
            elif outcome == -1: # Loss (Misere)
                self._returns[self._current_player] = -1.0
                self._returns[1 - self._current_player] = 1.0
        elif np.all(self._board != -1):
            self._game_over = True
            self._returns = [0.0, 0.0] # Draw
        
        if not self._game_over and self._moves_remaining_in_turn <= 0:
            self._current_player = 1 - self._current_player
            self._moves_remaining_in_turn = self._get_opening_moves(self._current_player)

    def _check_outcome(self, player):
        target_k = self._game.get_win_length(player)
        valid_dirs = self._game.get_valid_directions(player)
        is_misere = self._rules.get("misere", False)

        for r in range(self._rows):
            for c in range(self._cols):
                if self._board[r, c] != player: continue
                
                for dr, dc in valid_dirs:
                    if self._check_line(r, c, dr, dc, player, target_k):
                        return -1 if is_misere else 1
        return None

    def _check_line(self, r, c, dr, dc, player, k):
        for i in range(k):
            rr, cc = r + i * dr, c + i * dc
            if not (0 <= rr < self._rows and 0 <= cc < self._cols): return False
            if self._board[rr, cc] != player: return False
        return True

    def observation_tensor(self, player_id=None):
        # Note: pyspiel signature sometimes includes player_id
        obs = np.zeros((3, self._rows, self._cols))
        for r in range(self._rows):
            for c in range(self._cols):
                p = self._board[r, c]
                if p == 0: obs[0, r, c] = 1
                elif p == 1: obs[1, r, c] = 1
        return obs.flatten()

    def clone(self):
        new_state = MNKGameState(self._game, (self._rows, self._cols), self._rules)
        new_state._board = self._board.copy()
        new_state._current_player = self._current_player
        new_state._game_over = self._game_over
        new_state._returns = list(self._returns)
        new_state._move_count = self._move_count
        new_state._moves_remaining_in_turn = self._moves_remaining_in_turn
        return new_state
    
    def returns(self): return self._returns
    def is_terminal(self): return self._game_over
    
    # --- String Representation Helpers (Required for Sampler) ---
    def __str__(self):
        chars = {-1: '.', 0: 'X', 1: 'O'}
        return "\n".join("".join(chars[self._board[r, c]] for c in range(self._cols)) for r in range(self._rows))
    
    def to_string(self):
        return self.__str__()

    def action_to_string(self, player, action):
        row = action // self._cols
        col = action % self._cols
        return f"({row},{col})"


class MNKGame: # Removed pyspiel.Game
    def __init__(self, m=3, n=3, k=3, rules=None):
        self._m = m
        self._n = n
        self._base_k = k
        self._rules = rules if rules else {}
        
        # Validation
        k0 = self.get_win_length(0)
        k1 = self.get_win_length(1)
        max_k_needed = max(k0, k1)
        if min(m, n) < max_k_needed:
             if max(m, n) < max_k_needed:
                 raise ValueError(f"Grid size ({m}x{n}) too small for K={max_k_needed}")

    def new_initial_state(self):
        return MNKGameState(self, (self._m, self._n), self._rules)

    def num_players(self): return 2
    def max_game_length(self): return self._m * self._n
    def observation_tensor_shape(self): return (3, self._m, self._n)

    # --- Rule Helpers ---
    def get_win_length(self, player_id):
        extra = self._rules.get(f"p{player_id}_extra_k", 0)
        return self._base_k + extra

    def get_valid_directions(self, player_id):
        default_keys = ["h", "v", "d1", "d2"]
        allowed_map = self._rules.get("allowed_directions", {})
        keys = allowed_map.get(player_id, default_keys)
        mapping = {"h": (0, 1), "v": (1, 0), "d1": (1, 1), "d2": (1, -1)}
        return [mapping[k] for k in keys if k in mapping]