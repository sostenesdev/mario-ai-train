from math import ceil

from gymnasium import Wrapper
from shimmy import GymV21CompatibilityV0


class MarioGymnasiumEnv(GymV21CompatibilityV0):
    """Inclui a posicao inicial nas informacoes do reset da API antiga."""

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        info.update(self.gym_env.unwrapped._get_info())
        return observation, info


class HorizontalInactivityTimeout(Wrapper):
    """Encerra o episodio apos tres segundos de jogo sem movimento horizontal."""

    def __init__(self, env, seconds=3.0):
        super().__init__(env)
        self.fps = env.metadata.get(
            "render_fps", env.metadata.get("video.frames_per_second", 60)
        )
        if seconds <= 0 or self.fps <= 0:
            raise ValueError("O tempo limite e o FPS devem ser maiores que zero.")
        self.max_stationary_steps = ceil(seconds * self.fps)
        self._last_x = None
        self._stationary_steps = 0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._last_x = info["x_pos"]
        self._stationary_steps = 0
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        x_pos = info["x_pos"]
        if x_pos == self._last_x:
            self._stationary_steps += 1
        else:
            self._stationary_steps = 0
        self._last_x = x_pos

        if (
            self._stationary_steps >= self.max_stationary_steps
            and not terminated
            and not truncated
        ):
            truncated = True
            info = dict(info)
            info["episode_end_reason"] = "horizontal_inactivity"
            info["horizontal_inactivity_seconds"] = self._stationary_steps / self.fps

        return observation, reward, terminated, truncated, info
