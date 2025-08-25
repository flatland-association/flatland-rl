from flatland.core.effects_generator import EffectsGenerator


class MalfunctionEffectsGenerator(EffectsGenerator["RailEnv"]):

    def __init__(self, malfunction_generator):
        super().__init__()
        self.malfunction_generator = malfunction_generator

    def on_episode_step_start(self, env: "RailEnv", *args, **kwargs) -> "RailEnv":
        for agent in env.agents:
            agent.malfunction_handler.generate_malfunction(self.malfunction_generator, env.np_random)
        return env

# dummy
