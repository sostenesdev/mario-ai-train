import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

import main
from training_callbacks import BestModelCallback


class StoredModel:
    def __init__(self, label="initial"):
        self.label = label
        self.num_timesteps = 0
        self.device = "cpu"
        self.logger = Mock()

    def save(self, path):
        Path(path).write_text(self.label, encoding="utf-8")


class BestModelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "best_model.zip"
        self.model = StoredModel()
        self.callback = BestModelCallback(Mock(), self.path, eval_every=10)
        self.callback.init_callback(self.model)

    def test_worse_model_does_not_replace_best_and_improvement_does(self):
        with patch.object(
            self.callback, "_score", side_effect=[(100, 0), (50, 0), (120, 0)]
        ):
            self.callback.on_training_start({}, {})
            self.assertEqual(self.path.read_text(), "initial")
            self.model.label = "worse"
            self.model.num_timesteps = 10
            self.assertTrue(self.callback.on_step())
            self.assertEqual(self.path.read_text(), "initial")
            self.model.label = "better"
            self.model.num_timesteps = 20
            self.assertTrue(self.callback.on_step())
            self.assertEqual(self.path.read_text(), "better")

    def test_resume_rescores_saved_best_before_replacing_it(self):
        self.path.write_text("previous best")
        with patch(
            "training_callbacks.PPO.load", return_value=StoredModel("previous best")
        ) as load, patch.object(
            self.callback, "_score", side_effect=[(200, 0), (150, 0)]
        ):
            self.callback.on_training_start({}, {})
        load.assert_called_once()
        self.assertEqual(self.path.read_text(), "previous best")
        self.assertEqual(self.callback.best_mean_reward, 200)

    def test_training_end_evaluates_last_update_even_below_interval(self):
        with patch.object(
            self.callback, "_score", side_effect=[(100, 0), (120, 0)]
        ) as score:
            self.callback.on_training_start({}, {})
            self.model.label = "final"
            self.model.num_timesteps = 5
            self.callback.on_step()
            self.assertEqual(score.call_count, 1)
            self.callback.on_training_end()
            self.callback.on_training_end()
        self.assertEqual(score.call_count, 2)
        self.assertEqual(self.path.read_text(), "final")

    def test_failed_save_preserves_previous_best(self):
        self.path.write_text("previous best")
        self.callback.best_mean_reward = 100

        def interrupted_save(path):
            Path(path).write_text("incomplete")
            raise OSError("simulated save failure")

        with patch.object(self.callback, "_score", return_value=(120, 0)):
            with patch.object(self.model, "save", side_effect=interrupted_save):
                with self.assertRaises(OSError):
                    self.callback._evaluate()
        self.assertEqual(self.path.read_text(), "previous best")
        self.assertEqual(self.callback.best_mean_reward, 100)

    def test_evaluation_repeats_its_seed_and_preserves_training_rng(self):
        def score(*args, **kwargs):
            return torch.rand(1).item(), 0.0

        torch.manual_seed(123)
        state = torch.random.get_rng_state().clone()
        with patch("training_callbacks.evaluate_policy", side_effect=score):
            first = self.callback._score(self.model)
            second = self.callback._score(self.model)
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_playback_selects_best_with_latest_option(self):
        latest = self.path.with_name("latest.zip")
        with patch.object(main, "MODEL_PATH", latest):
            with patch.object(main, "BEST_MODEL_PATH", self.path):
                self.assertEqual(main.select_model_path(), latest)
                self.path.write_text("best")
                self.assertEqual(main.select_model_path(), self.path)
                self.assertEqual(main.select_model_path(latest=True), latest)

    def test_display_does_not_change_agent_observations_or_rewards(self):
        digests = []
        for display in (False, True):
            env = main.make_vec_env()
            digest = hashlib.sha256()
            try:
                env.seed(42)
                env.reset()
                with patch("cv2.imshow"), patch("cv2.waitKey", return_value=0):
                    for step in range(120):
                        action = np.array([4 if step % 60 < 30 else 3])
                        obs, rewards, dones, infos = env.step(action)
                        if display:
                            main.render_game(env)
                        digest.update(obs.tobytes())
                        digest.update(rewards.tobytes())
                        digest.update(dones.tobytes())
                digests.append(digest.hexdigest())
            finally:
                env.close()
        self.assertEqual(digests[0], digests[1])

    def test_evaluation_time_limit_is_reported_to_monitor(self):
        env = main.make_vec_env(max_episode_steps=10)
        try:
            env.reset()
            for _ in range(10):
                _, _, dones, infos = env.step(np.array([0]))
            self.assertTrue(dones[0])
            self.assertTrue(infos[0]["TimeLimit.truncated"])
            self.assertEqual(infos[0]["episode"]["l"], 10)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
