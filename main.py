import argparse
import os
import time
from collections import Counter
from pathlib import Path

import cv2
import torch
from gym_super_mario_bros import SuperMarioBrosEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, TimeLimit
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from wrappers import HorizontalInactivityTimeout, MarioGymnasiumEnv


MODEL_PATH = Path(__file__).with_name("mario_ppo.zip")
BEST_MODEL_PATH = MODEL_PATH.parent / "checkpoints" / "best_model.zip"


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Informe um inteiro maior que zero.")
    return number


def make_env():
    # Equivalente ao SuperMarioBros-v0, sem os wrappers da API nova do Gym.
    env = SuperMarioBrosEnv(rom_mode="vanilla")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = MarioGymnasiumEnv(env=env, render_mode="rgb_array")
    env = HorizontalInactivityTimeout(env, seconds=3.0)
    env = ResizeObservation(env, shape=84)
    env = GrayScaleObservation(env, keep_dim=True)
    return env


def make_vec_env(max_episode_steps=None):
    # Treinamento e execucao precisam do mesmo tratamento das observacoes.
    def create_env():
        env = make_env()
        if max_episode_steps is not None:
            env = TimeLimit(env, max_episode_steps=max_episode_steps)
        return Monitor(env)

    return VecFrameStack(DummyVecEnv([create_env]), n_stack=4)


def render_game(env):
    frame = env.render(mode="rgb_array")
    frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Mario AI Trainning", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)


def select_model_path(latest=False):
    if not latest and BEST_MODEL_PATH.is_file():
        return BEST_MODEL_PATH
    return MODEL_PATH


def play(steps=5000, deterministic=False, render=True, seed=None, latest=False):
    model_path = select_model_path(latest=latest)
    if not model_path.is_file():
        raise SystemExit(
            f"Modelo treinado nao encontrado: {model_path}. "
            "Execute primeiro: python train.py --steps 100000"
        )

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    env = make_vec_env()
    actions = Counter()
    max_x = 0
    try:
        model = PPO.load(model_path, env=env, device="cpu")
        print(f"Reproduzindo: {model_path}", flush=True)
        if seed is not None:
            torch.manual_seed(seed)
            env.seed(seed)
        obs = env.reset()
        mode = "deterministicas" if deterministic else "amostradas do modelo"
        print(f"Modelo: {model.num_timesteps} passos de treino. Acoes {mode}.", flush=True)
        print(
            "Esta execucao apenas reproduz o modelo. "
            "Para aprender: python train.py --steps 100000 --render",
            flush=True,
        )

        for step in range(1, steps + 1):
            action, _ = model.predict(obs, deterministic=deterministic)
            actions[int(action[0])] += 1
            # DummyVecEnv reinicia automaticamente os episodios finalizados.
            obs, rewards, dones, infos = env.step(action)
            max_x = max(max_x, infos[0]["x_pos"])
            if render:
                render_game(env)
                time.sleep(0.01)

            if infos[0].get("episode_end_reason") == "horizontal_inactivity":
                print(
                    "Episodio encerrado: 3 segundos sem deslocamento horizontal.",
                    flush=True,
                )

            if step % 300 == 0 or dones[0] or step == steps:
                print(
                    f"Passo {step}: x={infos[0]['x_pos']}, "
                    f"maior x={max_x}, botoes={SIMPLE_MOVEMENT[int(action[0])]}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\nReproducao interrompida.", flush=True)
    finally:
        env.close()
        if render:
            cv2.destroyAllWindows()

    print(f"Acoes executadas: {dict(actions)}", flush=True)
    return {"actions": dict(actions), "max_x": max_x}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproduz o modelo PPO do Mario.")
    parser.add_argument("--steps", type=positive_int, default=5000)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Escolhe sempre a acao mais provavel, para avaliar um modelo treinado.",
    )
    parser.add_argument("--headless", action="store_true", help="Executa sem abrir a janela.")
    parser.add_argument("--seed", type=int, help="Semente para reproduzir um teste.")
    parser.add_argument(
        "--latest", action="store_true",
        help="Reproduz o ultimo modelo salvo para comparar com a melhor versao.",
    )
    args = parser.parse_args(argv)
    return play(
        steps=args.steps,
        deterministic=args.deterministic,
        render=not args.headless,
        seed=args.seed,
        latest=args.latest,
    )


if __name__ == "__main__":
    main()
