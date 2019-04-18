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
