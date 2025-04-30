from pathlib import Path
from typing import Optional, List

from flatland.core.env import Environment


class FlatlandCallbacks:
    """
    Abstract base class for Flatland callbacks similar to rllib, see https://github.com/ray-project/ray/blob/master/rllib/callbacks/callbacks.py.

    These callbacks can be used for custom metrics and custom postprocessing.

    By default, all of these callbacks are no-ops.
    """

    def on_episode_start(
        self,
        *,
        env: Optional[Environment] = None,
        data_dir: Path = None,
        **kwargs,
    ) -> None:
        """Callback run right after an Episode has been started.

        This method gets called after `env.reset()`.

        Parameters
        ---------
            env : Environment
                the env
            data_dir : Path
                trajectory data dir
            kwargs:
                Forward compatibility placeholder.
        """
        pass

    def on_episode_step(
        self,
        *,
        env: Optional[Environment] = None,
        data_dir: Path = None,
        **kwargs,
    ) -> None:
        """Called on each episode step (after the action(s) has/have been logged).

        This callback is also called after the final step of an episode,
        meaning when terminated/truncated are returned as True
        from the `env.step()` call.

        The exact time of the call of this callback is after `env.step([action])` and
        also after the results of this step (observation, reward, terminated, truncated,
        infos) have been logged to the given `episode` object.

        Parameters
        ---------
            env : Environment
                the env
            data_dir : Path
                trajectory data dir
            kwargs:
                Forward compatibility placeholder.
        """
        pass

    def on_episode_end(
        self,
        *,
        env: Optional[Environment] = None,
        data_dir: Path = None,
        **kwargs,
    ) -> None:
        """Called when an episode is done (after terminated/truncated have been logged).

        The exact time of the call of this callback is after `env.step([action])`

        Parameters
        ---------
            env : Environment
                the env
            data_dir : Path
                trajectory data dir
            kwargs:
                Forward compatibility placeholder.
        """
        pass


# https://github.com/ray-project/ray/blob/3b94e5ff0038798a6955cde37459a0d30aa718c4/rllib/callbacks/utils.py#L41
def make_multi_callbacks(*_callback_list: FlatlandCallbacks):
    class _MultiFlatlandCallbacks(FlatlandCallbacks):
        IS_CALLBACK_CONTAINER = True

        def __init__(self, callback_list: List[FlatlandCallbacks]):
            self._callback_list = callback_list

        def on_episode_start(self, **kwargs) -> None:
            for callback in self._callback_list:
                callback.on_episode_start(**kwargs)

        def on_episode_step(self, **kwargs) -> None:
            for callback in self._callback_list:
                callback.on_episode_step(**kwargs)

        def on_episode_end(self, **kwargs) -> None:
            for callback in self._callback_list:
                callback.on_episode_end(**kwargs)

    return _MultiFlatlandCallbacks(_callback_list)
