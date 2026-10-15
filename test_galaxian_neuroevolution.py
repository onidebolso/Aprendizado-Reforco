import unittest
import csv
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from galaxian_neuroevolution import (
    Agente,
    carregar_agente,
    carregar_checkpoint,
    ConfiguracaoExperimento,
    criar_populacao,
    avaliar,
    avaliar_detalhado,
    mutar,
    nova_geracao,
    executar_experimento,
    salvar_resumo_experimento,
    selecionar,
    salvar_checkpoint,
)


class AmbienteSimulado:
    def __init__(self):
        self.action_space = EspacoSimulado()
        self.passos = 0

    def reset(self, seed=None):
        self.passos = 0
        return np.zeros(128, dtype=np.uint8), {}

    def step(self, acao):
        self.passos += 1
        terminou = self.passos == 2
        return np.zeros(128, dtype=np.uint8), 1.5, terminou, False, {}

    def close(self):
        pass


class EspacoSimulado:
    n = 3

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def sample(self):
        return int(self.rng.integers(self.n))


class TesteNeuroevolucao(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)

    def test_populacao_tem_individuos_independentes(self):
        populacao = criar_populacao(6, self.rng, tamanho=4)
        self.assertEqual(len(populacao), 4)
        self.assertEqual(populacao[0].W2.shape, (16, 6))
        self.assertFalse(np.array_equal(populacao[0].W1, populacao[1].W1))

    def test_agente_retorna_acao_valida(self):
        agente = Agente(6, self.rng)
        acao = agente.agir(np.zeros(128, dtype=np.uint8))
        self.assertGreaterEqual(acao, 0)
        self.assertLess(acao, 6)

    def test_agente_rejeita_ram_com_tamanho_incorreto(self):
        agente = Agente(6, self.rng)
        with self.assertRaises(ValueError):
            agente.agir(np.zeros(127, dtype=np.uint8))

    def test_carregar_agente_salvo(self):
        agente = Agente(6, self.rng)
        agente.fitness = 12.5
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "agente.npz"
            np.savez(
                caminho,
                W1=agente.W1,
                b1=agente.b1,
                W2=agente.W2,
                b2=agente.b2,
                fitness=agente.fitness,
            )
            carregado = carregar_agente(caminho, 6)
        np.testing.assert_array_equal(carregado.W1, agente.W1)
        np.testing.assert_array_equal(carregado.W2, agente.W2)
        self.assertEqual(carregado.fitness, 12.5)

    def test_avaliar_acumula_recompensas_ate_o_fim_do_episodio(self):
        agente = Agente(3, self.rng)
        env = AmbienteSimulado()
        fitness = avaliar(agente, env, seed=10, max_passos=10)
        self.assertEqual(fitness, 3.0)
        self.assertEqual(env.passos, 2)

    def test_avaliacao_detalhada_registra_fim_de_episodio(self):
        agente = Agente(3, self.rng)
        resultado = avaliar_detalhado(agente, AmbienteSimulado(), 10, 10)
        self.assertEqual(resultado.recompensa, 3.0)
        self.assertEqual(resultado.passos, 2)
        self.assertTrue(resultado.terminated)
        self.assertFalse(resultado.truncated)
        self.assertFalse(resultado.limite_atingido)

    def test_avaliacao_detalhada_identifica_limite_de_passos(self):
        class AmbienteSemFim(AmbienteSimulado):
            def step(self, acao):
                self.passos += 1
                return np.zeros(128, dtype=np.uint8), 1.0, False, False, {}

        resultado = avaliar_detalhado(
            Agente(3, self.rng), AmbienteSemFim(), 10, max_passos=1
        )
        self.assertEqual(resultado.passos, 1)
        self.assertFalse(resultado.terminated)
        self.assertFalse(resultado.truncated)
        self.assertTrue(resultado.limite_atingido)

    def test_checkpoint_restaura_populacao_historico_e_rng(self):
        populacao = criar_populacao(3, self.rng, tamanho=2)
        melhor = populacao[0]
        historico = [{"geracao": 0, "melhor": 1.0, "media": 0.5, "pior": 0.0}]
        registros = [{"tipo": "treino", "seed": 7}]
        metadados = {"seed": 7, "experimento": "A"}
        estado_rng = self.rng.bit_generator.state

        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "checkpoint.npz"
            salvar_checkpoint(
                caminho,
                populacao,
                melhor,
                1,
                historico,
                registros,
                self.rng,
                metadados,
            )
            rng_restaurado = np.random.default_rng(0)
            carregada, melhor_carregado, proxima, hist, regs = carregar_checkpoint(
                caminho, 3, rng_restaurado, metadados
            )

        self.assertEqual(proxima, 1)
        self.assertEqual(hist, historico)
        self.assertEqual(regs, registros)
        self.assertEqual(rng_restaurado.bit_generator.state, estado_rng)
        np.testing.assert_array_equal(carregada[1].W1, populacao[1].W1)
        np.testing.assert_array_equal(melhor_carregado.W2, melhor.W2)

    def test_experimento_curto_registra_episodios_e_pode_ser_retomado(self):
        def criar_env(**_kwargs):
            return AmbienteSimulado()

        with tempfile.TemporaryDirectory() as pasta:
            with patch("galaxian_neuroevolution.criar_ambiente", criar_env):
                primeira = executar_experimento(
                    configuracao=ConfiguracaoExperimento("A", 0.01, 0.05),
                    geracoes=1,
                    tamanho_populacao=2,
                    seed=11,
                    max_passos=5,
                    episodios_avaliacao=2,
                    pasta_saida=Path(pasta),
                    renderizar_melhor=False,
                    retomar=False,
                )
                retomada = executar_experimento(
                    configuracao=ConfiguracaoExperimento("A", 0.01, 0.05),
                    geracoes=2,
                    tamanho_populacao=2,
                    seed=11,
                    max_passos=5,
                    episodios_avaliacao=2,
                    pasta_saida=Path(pasta),
                    renderizar_melhor=False,
                    retomar=True,
                )
            episodios = (
                Path(pasta) / "episodios_A_seed_11.csv"
            ).read_text(encoding="utf-8")
            with (Path(pasta) / "episodios_A_seed_11.csv").open(
                newline="", encoding="utf-8"
            ) as arquivo:
                linhas_episodios = list(csv.DictReader(arquivo))

        self.assertEqual(len(primeira["historico"]), 1)
        self.assertEqual(len(retomada["historico"]), 2)
        self.assertIn("limite_atingido", episodios)
        self.assertIn("validacao", episodios)
        self.assertIn("aleatorio", episodios)
        self.assertEqual(len(linhas_episodios), 8)

    def test_resumo_agrega_media_e_desvio_entre_sementes(self):
        execucoes = [
            {
                "historico": [{"geracao": 0, "melhor": 2, "media": 1, "pior": 0}],
                "melhor_fitness": 2,
                "fitness_validacao": 2,
                "fitness_aleatorio": 1,
                "passos_validacao": 3,
                "taxa_limite_validacao": 0,
            },
            {
                "historico": [{"geracao": 0, "melhor": 4, "media": 2, "pior": 1}],
                "melhor_fitness": 4,
                "fitness_validacao": 4,
                "fitness_aleatorio": 2,
                "passos_validacao": 5,
                "taxa_limite_validacao": 1,
            },
        ]
        with tempfile.TemporaryDirectory() as pasta:
            resumo = salvar_resumo_experimento("A", execucoes, Path(pasta))
            with (Path(pasta) / "resumo_geracoes_A.csv").open(
                newline="", encoding="utf-8"
            ) as arquivo:
                geracao = next(csv.DictReader(arquivo))

        self.assertEqual(resumo["melhor_fitness_media"], 3.0)
        self.assertEqual(resumo["fitness_validacao_desvio_padrao"], 1.0)
        self.assertEqual(float(geracao["melhor_media"]), 3.0)
        self.assertEqual(float(geracao["melhor_desvio_padrao"]), 1.0)

    def test_selecao_prioriza_maior_fitness(self):
        populacao = criar_populacao(3, self.rng, tamanho=8)
        for indice, agente in enumerate(populacao):
            agente.fitness = float(indice)
        selecionados = selecionar(populacao)
        self.assertEqual(len(selecionados), 2)
        self.assertEqual([a.fitness for a in selecionados], [7.0, 6.0])

    def test_mutacao_zero_nao_altera_parametros(self):
        agente = Agente(4, self.rng)
        pesos_originais = agente.W1.copy()
        mutar(agente, self.rng, taxa_mutacao=0.0)
        np.testing.assert_array_equal(agente.W1, pesos_originais)

    def test_nova_geracao_preserva_elite_e_nao_compartilha_pesos(self):
        populacao = criar_populacao(4, self.rng, tamanho=6)
        for indice, agente in enumerate(populacao):
            agente.fitness = float(indice)
        elite_original = populacao[-1].W1.copy()

        proxima = nova_geracao(
            populacao,
            self.rng,
            taxa_mutacao=1.0,
            intensidade_mutacao=0.5,
            num_elite=1,
        )

        self.assertEqual(len(proxima), len(populacao))
        np.testing.assert_array_equal(proxima[0].W1, elite_original)
        self.assertFalse(np.shares_memory(proxima[1].W1, populacao[-1].W1))
        self.assertFalse(np.array_equal(proxima[1].W1, populacao[-1].W1))


if __name__ == "__main__":
    unittest.main()
