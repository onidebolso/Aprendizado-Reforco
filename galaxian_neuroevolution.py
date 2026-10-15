"""Evolucao de uma politica neural simples para o Atari Galaxian."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
STREAM_TREINO = 0
STREAM_VALIDACAO = 1
STREAM_EXIBICAO = 2

gym.register_envs(ale_py)


@dataclass(frozen=True)
class ConfiguracaoExperimento:
    nome: str
    taxa_mutacao: float
    intensidade_mutacao: float


@dataclass(frozen=True)
class ResultadoEpisodio:
    recompensa: float
    passos: int
    terminated: bool
    truncated: bool
    limite_atingido: bool


EXPERIMENTOS = {
    "A": ConfiguracaoExperimento("A", 0.01, 0.05),
    "B": ConfiguracaoExperimento("B", TAXA_MUTACAO, INTENSIDADE_MUTACAO),
    "C": ConfiguracaoExperimento("C", 0.40, 0.50),
}


def derivar_seed(seed: int, stream: int, indice: int) -> int:
    """Gera sementes deterministicas separadas por execucao, fluxo e episodio."""
    valor = int(
        np.random.SeedSequence([seed, stream, indice]).generate_state(
            1, dtype=np.uint32
        )[0]
    )
    return valor % (2**31 - 1)


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


def executar_episodio(
    escolher_acao: Callable[[np.ndarray], int],
    env,
    seed: int,
    max_passos: int = MAX_PASSOS_EPISODIO,
) -> ResultadoEpisodio:
    if max_passos < 1:
        raise ValueError("max_passos deve ser pelo menos 1.")
    estado, _ = env.reset(seed=seed)
    recompensa_total = 0.0
    terminated = False
    truncated = False
    passos = 0

    for _ in range(max_passos):
        acao = escolher_acao(estado)
        estado, recompensa, terminated, truncated, _ = env.step(acao)
        passos += 1
        recompensa_total += float(recompensa)
        if terminated or truncated:
            break

    return ResultadoEpisodio(
        recompensa=recompensa_total,
        passos=passos,
        terminated=bool(terminated),
        truncated=bool(truncated),
        limite_atingido=passos == max_passos and not terminated,
    )


def avaliar_detalhado(
    agente: Agente,
    env,
    seed: int,
    max_passos: int = MAX_PASSOS_EPISODIO,
) -> ResultadoEpisodio:
    """Executa um episodio e retorna recompensa e motivo de encerramento."""
    return executar_episodio(agente.agir, env, seed, max_passos)


def avaliar(
    agente: Agente,
    env,
    seed: int,
    max_passos: int = MAX_PASSOS_EPISODIO,
) -> float:
    """Executa um episodio e retorna a soma das recompensas como fitness."""
    return avaliar_detalhado(agente, env, seed, max_passos).recompensa


def avaliar_aleatorio(env, seed: int, max_passos: int = MAX_PASSOS_EPISODIO) -> float:
    """Executa o baseline que escolhe uniformemente entre as acoes disponiveis."""
    return avaliar_aleatorio_detalhado(env, seed, max_passos).recompensa


def avaliar_aleatorio_detalhado(
    env, seed: int, max_passos: int = MAX_PASSOS_EPISODIO
) -> ResultadoEpisodio:
    """Executa um episodio aleatorio e retorna metricas de encerramento."""
    if max_passos < 1:
        raise ValueError("max_passos deve ser pelo menos 1.")
    env.action_space.seed(seed)
    return executar_episodio(
        lambda _estado: env.action_space.sample(), env, seed, max_passos
    )


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


def salvar_agente(agente: Agente, caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        caminho,
        W1=agente.W1,
        b1=agente.b1,
        W2=agente.W2,
        b2=agente.b2,
        fitness=agente.fitness,
    )


def gravar_csv(caminho: Path, campos: tuple[str, ...], linhas: list[dict]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(linhas)


def registrar_episodio(
    registros: list[dict],
    tipo: str,
    resultado: ResultadoEpisodio,
    seed: int,
    geracao: int | None = None,
    individuo: int | None = None,
    episodio: int | None = None,
) -> None:
    registros.append(
        {
            "tipo": tipo,
            "geracao": geracao,
            "individuo": individuo,
            "episodio": episodio,
            "seed": seed,
            "recompensa": resultado.recompensa,
            "passos": resultado.passos,
            "terminated": resultado.terminated,
            "truncated": resultado.truncated,
            "limite_atingido": resultado.limite_atingido,
        }
    )


def salvar_checkpoint(
    caminho: Path,
    populacao: list[Agente],
    melhor_global: Agente,
    proxima_geracao: int,
    historico: list[dict[str, float | int]],
    registros: list[dict],
    rng: np.random.Generator,
    metadados: dict,
) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_name(f"{caminho.name}.tmp")
    with temporario.open("wb") as arquivo:
        np.savez_compressed(
            arquivo,
            W1=np.stack([agente.W1 for agente in populacao]),
            b1=np.stack([agente.b1 for agente in populacao]),
            W2=np.stack([agente.W2 for agente in populacao]),
            b2=np.stack([agente.b2 for agente in populacao]),
            fitness=np.asarray([agente.fitness for agente in populacao]),
            melhor_W1=melhor_global.W1,
            melhor_b1=melhor_global.b1,
            melhor_W2=melhor_global.W2,
            melhor_b2=melhor_global.b2,
            melhor_fitness=melhor_global.fitness,
            proxima_geracao=proxima_geracao,
            historico_json=json.dumps(historico),
            registros_json=json.dumps(registros),
            rng_json=json.dumps(rng.bit_generator.state),
            metadados_json=json.dumps(metadados, sort_keys=True),
        )
    temporario.replace(caminho)


def carregar_checkpoint(
    caminho: Path,
    num_acoes: int,
    rng: np.random.Generator,
    metadados_esperados: dict,
) -> tuple[list[Agente], Agente, int, list[dict], list[dict]]:
    with np.load(caminho, allow_pickle=False) as dados:
        metadados = json.loads(str(dados["metadados_json"]))
        if metadados != metadados_esperados:
            raise ValueError(
                "O checkpoint nao corresponde a semente, populacao, limite "
                "de passos ou configuracao deste experimento."
            )
        populacao = []
        for indice in range(len(dados["fitness"])):
            agente = Agente(num_acoes, rng)
            for nome in ("W1", "b1", "W2", "b2"):
                getattr(agente, nome)[:] = dados[nome][indice]
            agente.fitness = float(dados["fitness"][indice])
            populacao.append(agente)

        melhor = Agente(num_acoes, rng)
        for nome in ("W1", "b1", "W2", "b2"):
            getattr(melhor, nome)[:] = dados[f"melhor_{nome}"]
        melhor.fitness = float(dados["melhor_fitness"])
        rng.bit_generator.state = json.loads(str(dados["rng_json"]))
        return (
            populacao,
            melhor,
            int(dados["proxima_geracao"]),
            json.loads(str(dados["historico_json"])),
            json.loads(str(dados["registros_json"])),
        )


def salvar_resumo_experimento(
    nome: str,
    execucoes: list[dict],
    pasta: Path,
) -> dict:
    pasta.mkdir(parents=True, exist_ok=True)
    historicos = [execucao["historico"] for execucao in execucoes]
    geracoes = range(min(len(historico) for historico in historicos))
    resumo_geracoes = []
    for geracao in geracoes:
        linha: dict[str, float | int] = {"geracao": geracao}
        for metrica in ("melhor", "media", "pior"):
            valores = np.asarray(
                [historico[geracao][metrica] for historico in historicos],
                dtype=float,
            )
            linha[f"{metrica}_media"] = float(np.mean(valores))
            linha[f"{metrica}_desvio_padrao"] = float(np.std(valores))
        resumo_geracoes.append(linha)
    campos_geracoes = (
        "geracao",
        "melhor_media",
        "melhor_desvio_padrao",
        "media_media",
        "media_desvio_padrao",
        "pior_media",
        "pior_desvio_padrao",
    )
    gravar_csv(
        pasta / f"resumo_geracoes_{nome}.csv",
        campos_geracoes,
        resumo_geracoes,
    )

    metricas = (
        "melhor_fitness",
        "fitness_validacao",
        "fitness_aleatorio",
        "passos_validacao",
        "taxa_limite_validacao",
    )
    resumo: dict[str, str | int | float] = {
        "experimento": nome,
        "repeticoes": len(execucoes),
    }
    for metrica in metricas:
        valores = np.asarray([execucao[metrica] for execucao in execucoes], dtype=float)
        resumo[f"{metrica}_media"] = float(np.mean(valores))
        resumo[f"{metrica}_desvio_padrao"] = float(np.std(valores))

    if resumo_geracoes:
        x = [int(linha["geracao"]) for linha in resumo_geracoes]
        melhor = np.asarray([linha["melhor_media"] for linha in resumo_geracoes])
        desvio = np.asarray(
            [linha["melhor_desvio_padrao"] for linha in resumo_geracoes]
        )
        media = [float(linha["media_media"]) for linha in resumo_geracoes]
        plt.figure()
        plt.plot(x, melhor, label="Melhor fitness medio entre sementes")
        plt.fill_between(x, melhor - desvio, melhor + desvio, alpha=0.2)
        plt.plot(x, media, label="Fitness medio da populacao")
        plt.xlabel("Geracao")
        plt.ylabel("Fitness (soma das recompensas)")
        plt.title(f"Experimento {nome}: media e dispersao")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(pasta / f"experimento_{nome}.png", dpi=150)
        plt.close()
    return resumo


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
    episodios_avaliacao: int,
    pasta_saida: Path,
    renderizar_melhor: bool,
    retomar: bool,
) -> dict:
    rng = np.random.default_rng(seed)
    env = criar_ambiente(max_episode_steps=max_passos)
    nome_execucao = f"{configuracao.nome}_seed_{seed}"
    caminho_checkpoint = (
        pasta_saida / "checkpoints" / f"experimento_{nome_execucao}.npz"
    )
    metadados = {
        "experimento": configuracao.nome,
        "seed": seed,
        "populacao": tamanho_populacao,
        "max_passos": max_passos,
        "taxa_mutacao": configuracao.taxa_mutacao,
        "intensidade_mutacao": configuracao.intensidade_mutacao,
    }
    try:
        num_acoes = int(env.action_space.n)
        if retomar:
            if not caminho_checkpoint.exists():
                raise FileNotFoundError(
                    f"Nao foi encontrado checkpoint para retomar: "
                    f"{caminho_checkpoint}"
                )
            (
                populacao,
                melhor_global,
                primeira_geracao,
                historico,
                registros,
            ) = carregar_checkpoint(caminho_checkpoint, num_acoes, rng, metadados)
            if primeira_geracao > geracoes:
                raise ValueError(
                    f"O checkpoint ja alcancou {primeira_geracao} geracoes; "
                    f"--geracoes deve ser pelo menos esse valor."
                )
        else:
            populacao = criar_populacao(num_acoes, rng, tamanho_populacao)
            melhor_global = None
            primeira_geracao = 0
            historico = []
            registros = []
        print(
            f"\nExperimento {configuracao.nome}: "
            f"taxa={configuracao.taxa_mutacao:.0%}, "
            f"intensidade={configuracao.intensidade_mutacao:.2f}; "
            f"seed={seed}; "
            f"acoes disponiveis={num_acoes}"
        )

        for geracao in range(primeira_geracao, geracoes):
            seed_treino = derivar_seed(seed, STREAM_TREINO, geracao)
            for indice, agente in enumerate(populacao):
                # Todos os individuos da geracao enfrentam a mesma semente.
                resultado = avaliar_detalhado(
                    agente, env, seed_treino, max_passos=max_passos
                )
                agente.fitness = resultado.recompensa
                registrar_episodio(
                    registros,
                    "treino",
                    resultado,
                    seed_treino,
                    geracao=geracao,
                    individuo=indice,
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

            populacao = nova_geracao(
                populacao,
                rng,
                configuracao.taxa_mutacao,
                configuracao.intensidade_mutacao,
                num_elite=min(NUM_ELITE, len(populacao) - 1),
            )
            salvar_checkpoint(
                caminho_checkpoint,
                populacao,
                melhor_global,
                geracao + 1,
                historico,
                registros,
                rng,
                metadados,
            )

        if melhor_global is None:
            raise RuntimeError("Nenhum individuo foi avaliado.")

        validacoes = []
        baselines = []
        for episodio in range(episodios_avaliacao):
            seed_validacao = derivar_seed(seed, STREAM_VALIDACAO, episodio)
            resultado_agente = avaliar_detalhado(
                melhor_global, env, seed_validacao, max_passos=max_passos
            )
            resultado_aleatorio = avaliar_aleatorio_detalhado(
                env, seed_validacao, max_passos=max_passos
            )
            validacoes.append(resultado_agente)
            baselines.append(resultado_aleatorio)
            registrar_episodio(
                registros,
                "validacao",
                resultado_agente,
                seed_validacao,
                episodio=episodio,
            )
            registrar_episodio(
                registros,
                "aleatorio",
                resultado_aleatorio,
                seed_validacao,
                episodio=episodio,
            )

        fitness_validacao = float(np.mean([item.recompensa for item in validacoes]))
        fitness_aleatorio = float(np.mean([item.recompensa for item in baselines]))
        passos_validacao = float(np.mean([item.passos for item in validacoes]))
        taxa_limite = float(np.mean([item.limite_atingido for item in validacoes]))
        print(f"Fitness medio de validacao: {fitness_validacao:.2f}")
        print(f"Fitness medio aleatorio: {fitness_aleatorio:.2f}")
        print(f"Melhor fitness durante treino: {melhor_global.fitness:.2f}")

        pasta_saida.mkdir(parents=True, exist_ok=True)
        gravar_csv(
            pasta_saida / f"experimento_{nome_execucao}.csv",
            ("geracao", "melhor", "media", "pior"),
            historico,
        )
        gravar_csv(
            pasta_saida / f"episodios_{nome_execucao}.csv",
            (
                "tipo",
                "geracao",
                "individuo",
                "episodio",
                "seed",
                "recompensa",
                "passos",
                "terminated",
                "truncated",
                "limite_atingido",
            ),
            registros,
        )
        salvar_agente(
            melhor_global, pasta_saida / f"melhor_agente_{nome_execucao}.npz"
        )

        if renderizar_melhor:
            env.close()
            env = criar_ambiente(
                render_mode="human", max_episode_steps=max_passos
            )
            pontuacao = avaliar(
                melhor_global,
                env,
                derivar_seed(seed, STREAM_EXIBICAO, 0),
                max_passos=max_passos,
            )
            print(f"Pontuacao na exibicao do melhor agente: {pontuacao:.2f}")

        return {
            "seed": seed,
            "historico": historico,
            "melhor_fitness": melhor_global.fitness,
            "fitness_validacao": fitness_validacao,
            "fitness_aleatorio": fitness_aleatorio,
            "passos_validacao": passos_validacao,
            "taxa_limite_validacao": taxa_limite,
            "melhor_agente": melhor_global,
        }
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
    parser.add_argument(
        "--repeticoes",
        type=int,
        default=1,
        help="Numero de execucoes por experimento usando sementes consecutivas.",
    )
    parser.add_argument("--max-passos", type=int, default=MAX_PASSOS_EPISODIO)
    parser.add_argument(
        "--episodios-avaliacao",
        type=int,
        default=5,
        help="Episodios de validacao em sementes reservadas (padrao: 5).",
    )
    parser.add_argument("--saida", type=Path, default=Path("resultados"))
    parser.add_argument(
        "--retomar",
        action="store_true",
        help="Continua cada execucao a partir do checkpoint salvo.",
    )
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
    if args.seed < 0:
        parser.error("--seed deve ser nao negativo.")
    if args.assistir is not None:
        if args.episodios < 1:
            parser.error("--episodios deve ser pelo menos 1.")
        assistir_agente(args.assistir, args.episodios, args.seed, args.max_passos)
        return
    if args.geracoes < 1 or args.populacao < 2:
        parser.error("--geracoes deve ser positivo e --populacao deve ser >= 2.")
    if args.repeticoes < 1 or args.episodios_avaliacao < 1:
        parser.error("--repeticoes e --episodios-avaliacao devem ser positivos.")

    nomes = tuple(EXPERIMENTOS) if args.experimento == "all" else (args.experimento,)
    resumos = []
    linhas_execucoes = []
    for nome in nomes:
        execucoes = []
        for repeticao in range(args.repeticoes):
            seed = args.seed + repeticao
            execucao = executar_experimento(
                EXPERIMENTOS[nome],
                args.geracoes,
                args.populacao,
                seed,
                args.max_passos,
                args.episodios_avaliacao,
                args.saida,
                args.renderizar_melhor,
                args.retomar,
            )
            execucoes.append(execucao)
            linha = {
                campo: execucao[campo]
                for campo in (
                    "seed",
                    "melhor_fitness",
                    "fitness_validacao",
                    "fitness_aleatorio",
                    "passos_validacao",
                    "taxa_limite_validacao",
                )
            }
            linha["experimento"] = nome
            linhas_execucoes.append(linha)

        resumos.append(salvar_resumo_experimento(nome, execucoes, args.saida))
        melhor_execucao = max(execucoes, key=lambda item: item["fitness_validacao"])
        salvar_agente(
            melhor_execucao["melhor_agente"],
            args.saida / f"melhor_agente_{nome}.npz",
        )

    gravar_csv(
        args.saida / "resumo_experimentos.csv",
        tuple(resumos[0].keys()),
        resumos,
    )
    gravar_csv(
        args.saida / "resultados_por_semente.csv",
        (
            "experimento",
            "seed",
            "melhor_fitness",
            "fitness_validacao",
            "fitness_aleatorio",
            "passos_validacao",
            "taxa_limite_validacao",
        ),
        linhas_execucoes,
    )


if __name__ == "__main__":
    main()
