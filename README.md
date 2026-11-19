# Mario com PPO

Ative o ambiente virtual e instale as dependencias:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Treinar e acompanhar o aprendizado

```powershell
python train.py --steps 100000 --render
```

O Mario aprende enquanto o jogo aparece na janela. Para treinar sem a
janela, remova `--render`. O treinamento usa a CPU.

O arquivo `mario_ppo.zip` fica na mesma pasta de `main.py`. O progresso
e salvo a cada 10.000 passos, ao terminar e ao interromper com Ctrl+C.
Para alterar o intervalo, use `--save-every 5000`.

Se o arquivo ja existir, o treinamento continua a partir dele.
`--steps` indica quantos passos adicionais executar. PPO coleta blocos de
512 passos, portanto o total pode ultrapassar ligeiramente o solicitado.
A exploracao e incentivada tambem ao retomar modelos antigos.

Um treino curto serve para testar a geracao do arquivo. Aprender a jogar
exige treinamento muito mais longo; 100.000 passos nao garantem concluir
uma fase. Nao ha um modelo pre-treinado incluido nas dependencias.

## Assistir ao modelo salvo

```powershell
python main.py
```

Esta execucao nao treina nem altera o modelo. Por padrao, as acoes sao
amostradas das probabilidades aprendidas pelo PPO. Isso permite observar
as tentativas de um modelo ainda pouco treinado.

O modelo inicial de 2.048 passos escolhia apenas o botao de pulo quando
avaliado de forma deterministica, deixando o Mario na mesma posicao.
Para comparar esse modo ou avaliar um modelo mais treinado:

```powershell
python main.py --deterministic
```

O console informa os passos de treinamento do modelo, os botoes
executados e a posicao horizontal. Para um diagnostico sem janela:

```powershell
python main.py --headless --steps 600 --seed 42
```

O treinamento e a execucao compartilham imagens de 84x84 em tons de cinza
e quatro frames empilhados. A janela exibe o jogo na resolucao original.
O aviso de descontinuacao do Gym pode aparecer porque o emulador ainda o
utiliza internamente; a integracao com Gymnasium usa Shimmy.

## Limite de inatividade horizontal

O episodio termina automaticamente se a posicao horizontal do Mario nao
mudar durante 3 segundos de jogo (180 frames a 60 FPS). Isso vale tanto
para o treinamento quanto para a reproducao, com ou sem janela.

Qualquer deslocamento para a esquerda ou para a direita reinicia a
contagem. Pular sem sair do lugar nao reinicia. O limite usa o tempo do
jogo, independentemente da velocidade da CPU ou da renderizacao.

Ao atingir o limite, o episodio e encerrado por tempo limite
(`truncated=True`) e o ambiente vetorizado inicia outro automaticamente.
Na reproducao, o console informa o motivo do encerramento.
