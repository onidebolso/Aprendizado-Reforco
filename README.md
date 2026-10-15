# Neuroevolucao aplicada ao Atari Galaxian

Este projeto implementa um agente que aprende a jogar **Atari Galaxian** por
neuroevolucao. Ele foi desenvolvido para o trabalho pratico de Fundamentos de
Inteligencia Artificial e demonstra como uma politica de controle pode ser
otimizada por selecao e mutacao, sem ensinar ao agente a acao correta em cada
estado.

Em cada episodio, o jogo fornece ao agente uma observacao com 128 valores da
RAM do Atari. Uma rede neural pequena transforma esses valores em pontuacoes
para as acoes disponiveis; o agente escolhe a acao de maior pontuacao. A soma
das recompensas do episodio e o *fitness* usado para comparar os individuos.
Uma populacao de redes e avaliada, os individuos com melhor fitness sao
selecionados como pais, e novas geracoes sao criadas a partir de copias com
mutacoes aleatorias nos pesos e vieses. Os melhores individuos sao preservados
entre geracoes.

O objetivo e comparar como diferentes taxas e intensidades de mutacao afetam
o desempenho, observando as curvas de fitness e comparando o melhor agente com
uma estrategia de acoes aleatorias. O projeto usa Gymnasium e o Atari
Learning Environment (ALE), com o ambiente `ALE/Galaxian-v5`.

O metodo e uma busca evolutiva sobre os parametros da rede: **nao** usa DQN,
Q-Learning, exemplos rotulados, gradiente ou backpropagation.

## Adaptacao do enunciado

O material da atividade usa o Freeway, com tres acoes fixas. Este projeto usa
**Galaxian** (`ALE/Galaxian-v5`), conforme o jogo escolhido. A observacao
continua sendo a RAM do Atari (128 valores), mas o numero de acoes e obtido
diretamente de `env.action_space.n`; assim, a rede pode escolher entre as acoes
que o ALE disponibiliza para Galaxian, sem supor que sejam apenas `NOOP`, `UP`
e `DOWN`.

## Instalacao e configuracao

Recomendado: Python 3.10 ou mais recente.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Se a ativacao do ambiente virtual for bloqueada pelo PowerShell, use
`.\.venv\Scripts\python.exe` no lugar de `python`.

## Comandos de execucao

Os exemplos abaixo usam PowerShell e devem ser executados na pasta do projeto.
Depois de ativar o ambiente virtual conforme a secao anterior, treine com os
valores padrao (experimento B, 30 geracoes, populacao de 20 individuos):

```powershell
python galaxian_neuroevolution.py
```

O mesmo comando sem ativar o ambiente virtual:

```powershell
.\.venv\Scripts\python.exe galaxian_neuroevolution.py
```

Comparar os tres cenarios de mutacao em sequencia:

```powershell
python galaxian_neuroevolution.py --experimento all
```

Executar um cenario especifico:

```powershell
python galaxian_neuroevolution.py --experimento A
python galaxian_neuroevolution.py --experimento B
python galaxian_neuroevolution.py --experimento C
```

Fazer uma execucao curta para teste ou demonstracao:

```powershell
python galaxian_neuroevolution.py --experimento B --geracoes 3 --populacao 4 --max-passos 500 --renderizar-melhor
```

Assistir ao melhor agente de um treino salvo, sem treinar novamente:

```powershell
python galaxian_neuroevolution.py --assistir resultados\melhor_agente_B.npz
```

Assistir por tres episodios e limitar cada episodio a 5.000 passos:

```powershell
python galaxian_neuroevolution.py --assistir resultados\melhor_agente_B.npz --episodios 3 --max-passos 5000
```

`--renderizar-melhor` abre a janela do jogo depois de cada treino e mostra o
melhor agente encontrado naquele experimento. `--assistir` abre a janela
diretamente usando os pesos de um arquivo `.npz`. Para assistir, o arquivo
precisa existir; se ainda nao houver um agente salvo, execute primeiro um
comando de treino. O comando de ajuda lista as opcoes disponiveis:

```powershell
python galaxian_neuroevolution.py --help
```

Os testes automatizados podem ser executados com:

```powershell
python -m unittest -v test_galaxian_neuroevolution.py
```

Opcoes: `--experimento A|B|C|all`, `--geracoes`, `--populacao`, `--seed`,
`--max-passos`, `--saida`, `--renderizar-melhor`, `--assistir` e `--episodios`.
Os valores padrao sao: experimento B, 30 geracoes, populacao de 20 individuos,
semente 42, ate 5.000 passos por episodio e pasta de saida `resultados/`.
Use `--seed` para repetir ou variar a semente, `--saida` para escolher outra
pasta de resultados e `--max-passos` para alterar o limite de passos. Uma
populacao maior, mais geracoes ou episodios mais longos aumentam o tempo de
execucao.

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
