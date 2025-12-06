from state_sampler.base import StateSampler


class GameStageSampler(StateSampler):
    """
    Samples states from different stages of the game uniformly.
    """

    def __init__(self, game, max_free_space, min_free_space):
        super().__init__(game)
        self.max_free_space = max_free_space
        self.min_free_space = min_free_space

    def sample_state(self):
        # Placeholder implementation: returns the initial state
        return self.game.new_initial_state()
