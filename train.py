import argparse
import os

import cv2
import torch
from stable_baselines3 import PPO

from main import MODEL_PATH, make_vec_env, positive_int


# Incentiva a exploracao e limita mudancas bruscas da politica.
PPO_SETTINGS = {
    "n_steps": 512,
    "batch_size": 64,
    "n_epochs": 4,
    "ent_coef": 0.01,
    "target_kl": 0.02,
}


def train(steps, save_every=10000, render=False):
    # Muitas threads tornam as pequenas operacoes da CNN mais lentas na CPU.
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    env = make_vec_env()
    try:
        if MODEL_PATH.is_file():
            print(f"Continuando o treinamento de {MODEL_PATH}", flush=True)
            model = PPO.load(
                MODEL_PATH, env=env, device="cpu", **PPO_SETTINGS
            )
        else:
            print("Criando um novo modelo PPO com CnnPolicy.", flush=True)
            model = PPO(
                "CnnPolicy",
                env,
                learning_rate=2.5e-4,
                device="cpu",
                verbose=1,
                **PPO_SETTINGS,
            )

        last_save = model.num_timesteps

        def save_progress(local_vars, global_vars):
            nonlocal last_save
            if render:
                env.render(mode="human")
            if model.num_timesteps - last_save >= save_every:
                model.save(MODEL_PATH)
                last_save = model.num_timesteps
                print(f"Progresso salvo: {last_save} passos.", flush=True)
            return True

        display = "com a janela do jogo" if render else "sem abrir a janela do jogo"
        print(f"Treinando por mais {steps} passos, {display}.", flush=True)
        try:
            model.learn(
                total_timesteps=steps,
                callback=save_progress,
                reset_num_timesteps=False,
            )
        except KeyboardInterrupt:
            print("\nTreinamento interrompido. Salvando o progresso...", flush=True)

        model.save(MODEL_PATH)
        print(
            f"Modelo salvo em {MODEL_PATH} ({model.num_timesteps} passos acumulados).",
            flush=True,
        )
    finally:
        env.close()
        if render:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Treina o Mario e salva mario_ppo.zip na pasta do projeto."
    )
    parser.add_argument(
        "--steps",
        type=positive_int,
        default=100000,
        help="Passos adicionais de treinamento (padrao: 100000).",
    )
    parser.add_argument(
        "--save-every",
        type=positive_int,
        default=10000,
        help="Intervalo de salvamento em passos (padrao: 10000).",
    )
    parser.add_argument(
        "--render", action="store_true", help="Mostra o jogo durante o aprendizado."
    )
    args = parser.parse_args()
    train(args.steps, save_every=args.save_every, render=args.render)


if __name__ == "__main__":
    main()
