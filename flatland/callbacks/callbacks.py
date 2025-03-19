from typing import Optional

from flatland.core.env import Environment


class FlatlandCallbacks:
    """
    Abstract base class for Flatland callbacks similar to rllib, see https://github.com/ray-project/ray/blob/master/rllib/callbacks/callbacks.py.

    These callbacks can be used for custom metrics and custom postprocessing.

    By default, all of these callbacks are no-ops.
    """

    def on_episode_step(
        self,
        *,
        env: Optional[Environment] = None,
        **kwargs,
    ) -> None:
        """Called on each episode step (after the action(s) has/have been logged).

        This callback is also called after the final step of an episode,
        meaning when terminated/truncated are returned as True
        from the `env.step()` call.

        The exact time of the call of this callback is after `env.step([action])` and
        also after the results of this step (observation, reward, terminated, truncated,
        infos) have been logged to the given `episode` object.
        """
        pass
