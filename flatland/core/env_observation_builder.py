import numpy as np

# TODO: add docstrings, pylint, etc...


class ObservationBuilder:
    def __init__(self, env):
        self.env = env

    def reset(self):
        raise NotImplementedError()

    def get(self, handle):
        raise NotImplementedError()


class TreeObsForRailEnv(ObservationBuilder):
    def reset(self):
        # TODO: precompute distances, etc...
        # raise NotImplementedError()
        pass

    def get(self, handle):
        # TODO: compute the observation for agent `handle'

        # raise NotImplementedError()
        return []


class GlobalObsForRailEnv(ObservationBuilder):
    """
    Gives a global observation of the entire rail environment.
    The observation is composed of the following elements:

        - transition map array with dimensions (env.height, env.width, 16),
          assuming 16 bits encoding of transitions.

        - Four 2D arrays containing respectively the position of the given agent,
          the position of its target, the positions of the other agents and of
          their target.
    """
    def __init__(self, env):
        super(GlobalObsForRailEnv, self).__init__(env)
        self.rail_obs = np.zeros((self.env.height, self.env.width, 16))
        for i in range(self.rail_obs.shape[0]):
            for j in range(self.rail_obs.shape[1]):
                self.rail_obs[i, j] = self.env.rail.get_transitions((i, j))


