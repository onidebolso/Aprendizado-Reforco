"""Evolucao de uma politica neural simples para o Atari Galaxian."""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
from pathlib import Path

import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


ENV_ID = "ALE/Galaxian-v5"
NUM_ENTRADAS = 128
NUM_OCULTOS = 16
TAMANHO_POPULACAO = 20
NUM_GERACOES = 30
TAXA_MUTACAO = 0.10
INTENSIDADE_MUTACAO = 0.20
NUM_ELITE = 2
MAX_PASSOS_EPISODIO = 5_000

gym.register_envs(ale_py)


@dataclass(frozen=True)
class ConfiguracaoExperimento:
    nome: str
    taxa_mutacao: float
    intensidade_mutacao: float


EXPERIMENTOS = {
    "A": ConfiguracaoExperimento("A", 0.01, 0.05),
    "B": ConfiguracaoExperimento("B", TAXA_MUTACAO, INTENSIDADE_MUTACAO),
    "C": ConfiguracaoExperimento("C", 0.40, 0.50),
}


def criar_ambiente(
    render_mode: str | None = None,
    max_episode_steps: int = MAX_PASSOS_EPISODIO,
):
    """Cria Galaxian com observacao RAM de 128 bytes e acoes discretas ALE."""
    return gym.make(
        ENV_ID,
        obs_type="ram",
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
    )


class Agente:
    """Individuo cuja rede e codificada pelos pesos e vieses."""

    def __init__(self, num_acoes: int, rng: np.random.Generator):
        self.W1 = rng.standard_normal((NUM_ENTRADAS, NUM_OCULTOS))
        self.b1 = rng.standard_normal(NUM_OCULTOS)
        self.W2 = rng.standard_normal((NUM_OCULTOS, num_acoes))
        self.b2 = rng.standard_normal(num_acoes)
        self.fitness = 0.0

    def agir(self, estado: np.ndarray) -> int:
        estado = np.asarray(estado, dtype=np.float32).reshape(-1)
        if estado.size != NUM_ENTRADAS:
            raise ValueError(
                f"Esperados {NUM_ENTRADAS} valores de RAM; recebidos {estado.size}."
            )
        estado = estado / 255.0
        oculto = np.tanh(estado @ self.W1 + self.b1)
        saida = oculto @ self.W2 + self.b2
        return int(np.argmax(saida))


def avaliar(
    agente: Agente,
    env,
    seed: int,
    max_passos: int = MAX_PASSOS_EPISODIO,
) -> float:
    """Executa um episodio e retorna a soma das recompensas como fitness."""
    if max_passos < 1:
        raise ValueError("max_passos deve ser pelo menos 1.")
    estado, _ = env.reset(seed=seed)
    recompensa_total = 0.0

    for _ in range(max_passos):
        acao = agente.agir(estado)
        estado, recompensa, terminated, truncated, _ = env.step(acao)
        recompensa_total += float(recompensa)
        if terminated or truncated:
            break

    return recompensa_total


def avaliar_aleatorio(env, seed: int, max_passos: int = MAX_PASSOS_EPISODIO) -> float:
    """Executa o baseline que escolhe uniformemente entre as acoes disponiveis."""
    if max_passos < 1:
        raise ValueError("max_passos deve ser pelo menos 1.")
    env.action_space.seed(seed)
    estado, _ = env.reset(seed=seed)
    recompensa_total = 0.0

    for _ in range(max_passos):
        acao = env.action_space.sample()
        estado, recompensa, terminated, truncated, _ = env.step(acao)
        recompensa_total += float(recompensa)
        if terminated or truncated:
            break

    return recompensa_total


def criar_populacao(
    num_acoes: int,
    rng: np.random.Generator,
    tamanho: int = TAMANHO_POPULACAO,
) -> list[Agente]:
    if num_acoes < 1:
        raise ValueError("O ambiente deve disponibilizar pelo menos uma acao.")
    if tamanho < 2:
        raise ValueError("A populacao precisa ter pelo menos dois individuos.")
    return [Agente(num_acoes, rng) for _ in range(tamanho)]


def selecionar(
    populacao: list[Agente], proporcao: float = 0.25
) -> list[Agente]:
    """Seleciona os 25% de maior fitness (no minimo dois pais)."""
    if len(populacao) < 2:
        raise ValueError("Sao necessarios pelo menos dois individuos para selecionar.")
    if not 0.0 < proporcao <= 1.0:
        raise ValueError("A proporcao de selecao deve estar entre 0 e 1.")
    quantidade = min(len(populacao), max(2, int(np.ceil(len(populacao) * proporcao))))
    return sorted(populacao, key=lambda agente: agente.fitness, reverse=True)[
        :quantidade
    ]


def mutar(
    agente: Agente,
    rng: np.random.Generator,
    taxa_mutacao: float = TAXA_MUTACAO,
    intensidade_mutacao: float = INTENSIDADE_MUTACAO,
) -> None:
    """Soma ruido gaussiano a cada parametro com a probabilidade configurada."""
    if not 0.0 <= taxa_mutacao <= 1.0:
        raise ValueError("A taxa de mutacao deve estar entre 0 e 1.")
    if intensidade_mutacao < 0.0:
        raise ValueError("A intensidade da mutacao nao pode ser negativa.")
    for parametros in (agente.W1, agente.b1, agente.W2, agente.b2):
        mascara = rng.random(parametros.shape) < taxa_mutacao
        parametros[mascara] += rng.normal(
            0.0, intensidade_mutacao, size=int(mascara.sum())
        )


def nova_geracao(
    populacao: list[Agente],
    rng: np.random.Generator,
    taxa_mutacao: float = TAXA_MUTACAO,
    intensidade_mutacao: float = INTENSIDADE_MUTACAO,
    num_elite: int = NUM_ELITE,
) -> list[Agente]:
    """Preserva os melhores e completa a populacao com copias mutadas dos pais."""
    if not populacao:
        raise ValueError("A populacao nao pode estar vazia.")
    if not 1 <= num_elite < len(populacao):
        raise ValueError("NUM_ELITE deve ser pelo menos 1 e menor que a populacao.")
    pais = selecionar(populacao)
    ordenada = sorted(populacao, key=lambda agente: agente.fitness, reverse=True)
    proxima = [copy.deepcopy(agente) for agente in ordenada[:num_elite]]

    while len(proxima) < len(populacao):
        filho = copy.deepcopy(pais[int(rng.integers(len(pais)))])
        mutar(filho, rng, taxa_mutacao, intensidade_mutacao)
        filho.fitness = 0.0
        proxima.append(filho)

    return proxima


def salvar_resultados(
    historico: list[dict[str, float | int]],
    melhor_agente: Agente,
    pasta: Path,
    nome: str,
) -> None:
    pasta.mkdir(parents=True, exist_ok=True)
    caminho_csv = pasta / f"experimento_{nome}.csv"
    with caminho_csv.open("w", newline="", encoding="utf-8") as arquivo:
        campos = ("geracao", "melhor", "media", "pior")
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(historico)

    geracoes = [int(item["geracao"]) for item in historico]
    melhores = [float(item["melhor"]) for item in historico]
    medias = [float(item["media"]) for item in historico]
    plt.figure()
    plt.plot(geracoes, melhores, label="Melhor fitness")
    plt.plot(geracoes, medias, label="Fitness medio")
    plt.xlabel("Geracao")
    plt.ylabel("Fitness (soma das recompensas)")
    plt.title(f"Evolucao - Experimento {nome}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(pasta / f"experimento_{nome}.png", dpi=150)
    plt.close()

    np.savez(
        pasta / f"melhor_agente_{nome}.npz",
        W1=melhor_agente.W1,
        b1=melhor_agente.b1,
        W2=melhor_agente.W2,
        b2=melhor_agente.b2,
        fitness=melhor_agente.fitness,
    )


def carregar_agente(caminho: Path, num_acoes: int) -> Agente:
    """Carrega pesos salvos e verifica se correspondem ao ambiente atual."""
    with np.load(caminho, allow_pickle=False) as dados:
        campos = ("W1", "b1", "W2", "b2", "fitness")
        ausentes = [campo for campo in campos if campo not in dados]
        if ausentes:
            raise ValueError(
                f"Arquivo de agente invalido; campos ausentes: {', '.join(ausentes)}."
            )

        agente = Agente(num_acoes, np.random.default_rng(0))
        parametros = {
            "W1": (NUM_ENTRADAS, NUM_OCULTOS),
            "b1": (NUM_OCULTOS,),
            "W2": (NUM_OCULTOS, num_acoes),
            "b2": (num_acoes,),
        }
        for nome, formato in parametros.items():
            valor = dados[nome]
            if valor.shape != formato:
                raise ValueError(
                    f"Formato invalido para {nome}: esperado {formato}, "
                    f"recebido {valor.shape}."
                )
            getattr(agente, nome)[:] = valor
        agente.fitness = float(dados["fitness"])
        return agente


def assistir_agente(
    caminho: Path,
    episodios: int,
    seed: int,
    max_passos: int,
) -> None:
    """Abre uma janela e mostra um agente salvo jogando."""
    env = criar_ambiente(
        render_mode="human", max_episode_steps=max_passos
    )
    try:
        agente = carregar_agente(caminho, int(env.action_space.n))
        print(
            f"Assistindo agente: {caminho} "
            f"(fitness salvo: {agente.fitness:.2f})"
        )
        for numero in range(episodios):
            pontuacao = avaliar(
                agente, env, seed + numero, max_passos=max_passos
            )
            print(f"Episodio {numero + 1}/{episodios} | Recompensa: {pontuacao:.2f}")
    finally:
        env.close()


def executar_experimento(
    configuracao: ConfiguracaoExperimento,
    geracoes: int,
    tamanho_populacao: int,
    seed: int,
    max_passos: int,
    pasta_saida: Path,
    renderizar_melhor: bool,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    env = criar_ambiente(max_episode_steps=max_passos)
    try:
        num_acoes = int(env.action_space.n)
        populacao = criar_populacao(num_acoes, rng, tamanho_populacao)
        melhor_global: Agente | None = None
        historico: list[dict[str, float | int]] = []
        print(
            f"\nExperimento {configuracao.nome}: "
            f"taxa={configuracao.taxa_mutacao:.0%}, "
            f"intensidade={configuracao.intensidade_mutacao:.2f}; "
            f"acoes disponiveis={num_acoes}"
        )

        for geracao in range(geracoes):
            for agente in populacao:
                # Todos os individuos da geracao enfrentam a mesma semente.
                agente.fitness = avaliar(
                    agente, env, seed + geracao, max_passos=max_passos
                )
                if melhor_global is None or agente.fitness > melhor_global.fitness:
                    melhor_global = copy.deepcopy(agente)

            valores = [agente.fitness for agente in populacao]
            melhor = max(valores)
            media = float(np.mean(valores))
            pior = min(valores)
            historico.append(
                {
                    "geracao": geracao,
                    "melhor": melhor,
                    "media": media,
                    "pior": pior,
                }
            )
            print(
                f"Geracao {geracao:03d} | "
                f"Melhor: {melhor:.2f} | Media: {media:.2f} | Pior: {pior:.2f}"
            )

            if geracao < geracoes - 1:
                populacao = nova_geracao(
                    populacao,
                    rng,
                    configuracao.taxa_mutacao,
                    configuracao.intensidade_mutacao,
                    num_elite=min(NUM_ELITE, len(populacao) - 1),
                )

        if melhor_global is None:
            raise RuntimeError("Nenhum individuo foi avaliado.")

        baseline = avaliar_aleatorio(
            env, seed + geracoes, max_passos=max_passos
        )
        print(f"Baseline aleatorio: {baseline:.2f}")
        print(f"Melhor agente evoluido: {melhor_global.fitness:.2f}")
        salvar_resultados(historico, melhor_global, pasta_saida, configuracao.nome)

        if renderizar_melhor:
            env.close()
            env = criar_ambiente(
                render_mode="human", max_episode_steps=max_passos
            )
            pontuacao = avaliar(
                melhor_global, env, seed + geracoes + 1, max_passos=max_passos
            )
            print(f"Pontuacao na exibicao do melhor agente: {pontuacao:.2f}")

        return melhor_global.fitness, baseline
    finally:
        env.close()


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evolui uma rede neural para jogar Atari Galaxian."
    )
    parser.add_argument(
        "--experimento",
        choices=("A", "B", "C", "all"),
        default="B",
        help="A/B/C compara mutacoes sugeridas; all executa as tres.",
    )
    parser.add_argument("--geracoes", type=int, default=NUM_GERACOES)
    parser.add_argument("--populacao", type=int, default=TAMANHO_POPULACAO)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-passos", type=int, default=MAX_PASSOS_EPISODIO)
    parser.add_argument("--saida", type=Path, default=Path("resultados"))
    parser.add_argument(
        "--assistir",
        type=Path,
        metavar="ARQUIVO_NPZ",
        help="Carrega um agente salvo e abre o Galaxian para assisti-lo jogar.",
    )
    parser.add_argument(
        "--episodios",
        type=int,
        default=1,
        help="Numero de episodios ao usar --assistir (padrao: 1).",
    )
    parser.add_argument(
        "--renderizar-melhor",
        action="store_true",
        help="Abre a janela do jogo e executa o melhor agente de cada experimento.",
    )
    return parser


def main() -> None:
    parser = criar_parser()
    args = parser.parse_args()
    if args.max_passos < 1:
        parser.error("--max-passos deve ser positivo.")
    if args.assistir is not None:
        if args.episodios < 1:
            parser.error("--episodios deve ser pelo menos 1.")
        assistir_agente(args.assistir, args.episodios, args.seed, args.max_passos)
        return
    if args.geracoes < 1 or args.populacao < 2:
        parser.error("--geracoes deve ser positivo e --populacao deve ser >= 2.")

    nomes = tuple(EXPERIMENTOS) if args.experimento == "all" else (args.experimento,)
    for nome in nomes:
        executar_experimento(
            EXPERIMENTOS[nome],
            args.geracoes,
            args.populacao,
            args.seed,
            args.max_passos,
            args.saida,
            args.renderizar_melhor,
        )


if __name__ == "__main__":
    main()
