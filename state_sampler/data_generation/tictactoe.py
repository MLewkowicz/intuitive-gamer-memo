import pyspiel
import numpy as np

game = pyspiel.load_game("tic_tac_toe")

DIRECTIONS = [(1,0),(0,1),(1,1),(1,-1)]  # h,v,diag,anti

def longest_chain(board, player):
    """Return max contiguous stones for `player` on board."""
    best = 0
    rows, cols = board.shape
    for r in range(rows):
        for c in range(cols):
            if board[r, c] != player: 
                continue
            for dr, dc in DIRECTIONS:
                length = 1
                rr, cc = r + dr, c + dc
                while 0 <= rr < rows and 0 <= cc < cols and board[rr, cc] == player:
                    length += 1
                    rr += dr; cc += dc
                best = max(best, length)
    return best


def generate_all_states(state, visited):
    # add current state
    key = state.to_string()       # or any hashable representation
    if key in visited:
        return
    visited[key] = (state, len(state.legal_actions()))

    me = 1 if state.current_player() == 0 else -1
    opp = -me
    

    # if this is terminal we stop after adding
    if state.is_terminal():
        return

    # otherwise expand
    for a in state.legal_actions():
        child = state.clone()     # IMPORTANT!
        child.apply_action(a)
        generate_all_states(child, visited)

all_states = {}
generate_all_states(game.new_initial_state(), all_states)

print(f"Total states generated: {len(all_states)}")