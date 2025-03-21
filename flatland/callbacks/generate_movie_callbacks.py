"""
FlatlandCallbacks for rendering the env after each step and compiling movie when the episode is done.
"""

import os
from pathlib import Path
from typing import Optional

from flatland.callbacks.callbacks import FlatlandCallbacks
from flatland.envs.rail_env import RailEnv
from flatland.evaluators import aicrowd_helpers
from flatland.utils.rendertools import RenderTool


class GenerateMovieCallbacks(FlatlandCallbacks):
    def __init__(self):
        self.renderer: RenderTool = None

    def on_episode_start(
        self,
        *,
        env: Optional[RailEnv] = None,
        data_dir: Path = None,
        **kwargs,
    ) -> None:
        self.renderer = RenderTool(env, gl="PILSVG")
        self._render_env(0, data_dir)

    def on_episode_step(
        self,
        *,
        env: Optional[RailEnv] = None,
        data_dir: Path = None,
        **kwargs,
    ) -> None:
        self._render_env(env._elapsed_steps, data_dir)

    def on_episode_end(
        self,
        *,
        env: Optional[RailEnv] = None,
        data_dir: Path = None,
        **kwargs,
    ) -> None:
        video_output_path, video_thumb_output_path = \
            aicrowd_helpers.generate_movie_from_frames(
                data_dir
            )
        print("Videos : ", video_output_path, video_thumb_output_path)

    def _render_env(self, env_time: int, data_dir: Path):
        self.renderer.render_env(show=False,
                                 show_observations=False,
                                 show_predictions=False,
                                 show_rowcols=False)
        self.renderer.gl.save_image(
            os.path.join(
                data_dir,
                "flatland_frame_{:04d}.png".format(env_time)
            ))
