import unittest

import gymnasium as gym
import numpy as np

from main import make_env, make_vec_env
from wrappers import HorizontalInactivityTimeout


class PositionEnv(gym.Env):
    metadata = {"render_fps": 60}

    def __init__(self, end_at=None, truncate_at=None):
        self.observation_space = gym.spaces.Box(0, 255, shape=(1,), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(3)
        self.end_at = end_at
        self.truncate_at = truncate_at

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.x = 40
        self.steps = 0
        return np.zeros(1, dtype=np.uint8), {"x_pos": self.x, "y_pos": 79}

    def step(self, action):
        self.steps += 1
        self.x += {0: 0, 1: 1, 2: -1}[action]
        return (
            np.zeros(1, dtype=np.uint8),
            2.0,
            self.steps == self.end_at,
            self.steps == self.truncate_at,
            {"x_pos": self.x, "y_pos": 79 + self.steps % 30},
        )


class HorizontalInactivityTests(unittest.TestCase):
    def test_stops_at_exactly_180_frames_even_with_vertical_movement(self):
        env = HorizontalInactivityTimeout(PositionEnv())
        env.reset()
        for _ in range(179):
            _, reward, terminated, truncated, _ = env.step(0)
            self.assertFalse(terminated or truncated)
        _, reward, terminated, truncated, info = env.step(0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(reward, 2.0)
        self.assertEqual(info["episode_end_reason"], "horizontal_inactivity")
        self.assertEqual(info["horizontal_inactivity_seconds"], 3.0)

    def test_movement_in_either_direction_restarts_the_timer(self):
        for action in (1, 2):
            with self.subTest(action=action):
                env = HorizontalInactivityTimeout(PositionEnv())
                env.reset()
                for _ in range(179):
                    env.step(0)
                self.assertFalse(env.step(action)[3])
                for _ in range(179):
                    self.assertFalse(env.step(0)[3])
                self.assertTrue(env.step(0)[3])

    def test_continuous_movement_does_not_end_episode(self):
        env = HorizontalInactivityTimeout(PositionEnv())
        env.reset()
        for action in [1, 2] * 200:
            _, _, terminated, truncated, _ = env.step(action)
            self.assertFalse(terminated or truncated)

    def test_reset_clears_previous_inactivity(self):
        env = HorizontalInactivityTimeout(PositionEnv())
        env.reset()
        for _ in range(180):
            env.step(0)
        env.reset()
        for _ in range(179):
            self.assertFalse(env.step(0)[3])
        self.assertTrue(env.step(0)[3])

    def test_preserves_natural_episode_end(self):
        env = HorizontalInactivityTimeout(PositionEnv(end_at=180))
        env.reset()
        for _ in range(180):
            result = env.step(0)
        self.assertTrue(result[2])
        self.assertFalse(result[3])
        self.assertNotIn("episode_end_reason", result[4])

    def test_preserves_existing_truncation(self):
        env = HorizontalInactivityTimeout(PositionEnv(truncate_at=180))
        env.reset()
        for _ in range(180):
            result = env.step(0)
        self.assertFalse(result[2])
        self.assertTrue(result[3])
        self.assertNotIn("episode_end_reason", result[4])

    def test_uses_environment_frame_rate(self):
        base = PositionEnv()
        base.metadata = {"video.frames_per_second": 30}
        env = HorizontalInactivityTimeout(base)
        env.reset()
        for _ in range(89):
            self.assertFalse(env.step(0)[3])
        self.assertTrue(env.step(0)[3])

    def test_real_mario_reset_and_exact_timeout(self):
        env = make_env()
        try:
            obs, info = env.reset()
            initial_x = info["x_pos"]
            self.assertEqual(obs.shape, (84, 84, 1))
            for _ in range(179):
                _, _, terminated, truncated, info = env.step(0)
                self.assertEqual(info["x_pos"], initial_x)
                self.assertFalse(terminated or truncated)
            _, _, terminated, truncated, info = env.step(0)
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(info["episode_end_reason"], "horizontal_inactivity")
        finally:
            env.close()

    def test_vector_env_restarts_two_episodes_and_clears_frame_stack(self):
        env = make_vec_env()
        try:
            env.reset()
            ended_at = []
            for step in range(1, 361):
                obs, _, dones, infos = env.step(np.array([0]))
                if dones[0]:
                    ended_at.append(step)
                    self.assertTrue(infos[0]["TimeLimit.truncated"])
                    self.assertEqual(infos[0]["episode"]["l"], 180)
                    self.assertEqual(
                        infos[0]["terminal_observation"].shape, (84, 84, 4)
                    )
                    self.assertEqual(
                        infos[0]["episode_end_reason"], "horizontal_inactivity"
                    )
                    self.assertTrue(np.all(obs[0, :, :, :3] == 0))
            self.assertEqual(ended_at, [180, 360])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
