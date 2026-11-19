from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


class BestModelCallback(BaseCallback):
    """Guarda o melhor resultado de uma avaliacao separada do treinamento."""

    def __init__(
        self,
        eval_env,
        best_model_path,
        eval_every=25000,
        n_eval_episodes=5,
        seed=42,
        process_events=None,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.best_model_path = Path(best_model_path)
        self.eval_every = eval_every
        self.n_eval_episodes = n_eval_episodes
        self.seed = seed
        self.process_events = process_events
        self.best_mean_reward = float("-inf")
        self._last_evaluation_step = None

    def _score(self, model):
        # A avaliacao usa sua propria sequencia aleatoria e preserva a do treino.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.eval_env.seed(self.seed)
            return evaluate_policy(
                model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=False,
                callback=self._process_events if self.process_events else None,
            )

    def _process_events(self, local_vars, global_vars):
        self.process_events()

    def _on_training_start(self):
        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
        if self.best_model_path.is_file():
            print("Avaliando a melhor versao salva para comparar com o treino atual...", flush=True)
            with torch.random.fork_rng(devices=[]):
                previous = PPO.load(
                    self.best_model_path, env=self.eval_env, device=self.model.device
                )
                self.best_mean_reward, _ = self._score(previous)
                del previous
        self._evaluate()

    def _evaluate(self):
        print(
            f"Avaliando {self.n_eval_episodes} episodios separados "
            f"({self.model.num_timesteps} passos de treino)...",
            flush=True,
        )
        mean_reward, std_reward = self._score(self.model)
        if mean_reward > self.best_mean_reward:
            temporary = self.best_model_path.with_suffix(".tmp.zip")
            self.model.save(temporary)
            temporary.replace(self.best_model_path)
            self.best_mean_reward = float(mean_reward)
            print(f"Melhor modelo salvo em {self.best_model_path}", flush=True)

        self._last_evaluation_step = self.model.num_timesteps
        self.logger.record("eval/mean_reward", float(mean_reward))
        self.logger.record("eval/best_mean_reward", self.best_mean_reward)
        print(
            f"Recompensa media: {mean_reward:.2f} (+/- {std_reward:.2f}); "
            f"melhor: {self.best_mean_reward:.2f}.",
            flush=True,
        )

    def _on_step(self):
        if self.model.num_timesteps - self._last_evaluation_step >= self.eval_every:
            self._evaluate()
        return True

    def _on_training_end(self):
        if self.model.num_timesteps != self._last_evaluation_step:
            self._evaluate()
