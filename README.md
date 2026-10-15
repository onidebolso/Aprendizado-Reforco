# Neuroevolucao aplicada ao Atari Galaxian

Projeto para o trabalho pratico de Fundamentos de Inteligencia Artificial. O
agente aprende por algoritmo evolutivo: nao usa DQN, Q-Learning, exemplos
rotulados nem backpropagation.

## Adaptacao do enunciado

O material da atividade usa o Freeway, com tres acoes fixas. Este projeto usa
**Galaxian** (`ALE/Galaxian-v5`), conforme o jogo escolhido. A observacao
continua sendo a RAM do Atari (128 valores), mas o numero de acoes e obtido
diretamente de `env.action_space.n`; assim, a rede pode escolher entre as acoes
que o ALE disponibiliza para Galaxian, sem supor que sejam apenas `NOOP`, `UP`
e `DOWN`.

## Preparacao

Recomendado: Python 3.10 ou mais recente.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Se a ativacao do ambiente virtual for bloqueada pelo PowerShell, use
`.\.venv\Scripts\python.exe` no lugar de `python`.

## Executar

O experimento B (taxa de mutacao de 10% e intensidade 0,20) e o padrao:

```powershell
python galaxian_neuroevolution.py
```

Comparar os tres cenarios sugeridos no enunciado (A: 1%/0,05; B: 10%/0,20;
C: 40%/0,50):

```powershell
python galaxian_neuroevolution.py --experimento all
```

Para uma execucao curta de teste ou demonstracao:

```powershell
python galaxian_neuroevolution.py --experimento B --geracoes 3 --populacao 4 --max-passos 500 --renderizar-melhor
```

Para assistir a um agente que ja foi treinado e salvo, sem treinar novamente:

```powershell
python galaxian_neuroevolution.py --assistir resultados\melhor_agente_B.npz
```

Para mostrar mais episodios ou definir por quanto tempo o jogo pode rodar:

```powershell
python galaxian_neuroevolution.py --assistir resultados\melhor_agente_B.npz --episodios 3 --max-passos 5000
```

`--renderizar-melhor` abre a janela depois do treino; `--assistir` abre a janela
diretamente com os pesos de um arquivo `.npz`. Se ainda nao houver um agente
salvo, execute primeiro um dos comandos de treino acima. A janela pode ser
fechada normalmente ao terminar o episodio.

Opcoes: `--experimento A|B|C|all`, `--geracoes`, `--populacao`, `--seed`,
`--max-passos`, `--saida`, `--renderizar-melhor`, `--assistir` e `--episodios`.
Cada episodio e limitado a 5.000 passos por padrao para manter o custo de treino
finito; altere o limite com `--max-passos`.

## O que o programa implementa

- Rede neural `128 -> 16 -> N`, em que `N` e o numero de acoes de Galaxian.
  A funcao `Agente.agir` normaliza os valores da RAM, calcula `tanh` na camada
  oculta e escolhe a acao com maior saida (`argmax`).
- `avaliar` reinicia o ambiente, chama `env.step(acao)`, acumula as recompensas
  e encerra o episodio quando `terminated` ou `truncated` for verdadeiro. A
  soma das recompensas e usada como fitness.
- Populacao inicial aleatoria, selecao dos 25% melhores (minimo de dois pais),
  copia sem alteracao dos dois melhores (elitismo) e filhos criados por copia
  independente de um pai, seguida de mutacao gaussiana dos pesos e vieses.
- Fitness melhor, medio e pior registrados para cada geracao.
- Baseline aleatorio executado para comparacao com o melhor individuo evoluido.

## Arquivos gerados

Na pasta `resultados/`, cada experimento salva:

- `experimento_X.csv`: fitness melhor, medio e pior por geracao;
- `experimento_X.png`: grafico do melhor e do fitness medio;
- `melhor_agente_X.npz`: pesos e vieses do melhor agente encontrado.

## Como explicar e analisar

**Recompensa** e o sinal dado pelo jogo a cada acao; **fitness** e a medida
usada para comparar individuos e, neste projeto, corresponde a soma das
recompensas de um episodio. Todos os individuos de uma geracao usam a mesma
semente para reduzir diferencas causadas pela aleatoriedade do ambiente.

A exploracao vem dos pesos iniciais aleatorios e das mutacoes. No cenario A,
as mudancas pequenas podem manter comportamentos bons, mas explorar lentamente.
No cenario C, surgem mudancas maiores e mais diversidade, com maior chance de
destruir parametros uteis. O grafico e o CSV permitem observar se o melhor e o
fitness medio continuam subindo ou se estabilizam (convergencia). Compare
tambem o melhor fitness evoluido com o baseline aleatorio.

Ao executar `--experimento all`, os tres cenarios usam a mesma semente para
partir dos mesmos pesos iniciais e sequencias de ambiente. Os resultados de uma
unica execucao ainda podem variar. Para uma conclusao mais confiavel, rode os
cenarios mais de uma vez mudando `--seed` e compare as curvas, sem esquecer de
informar esse procedimento no relatorio.
