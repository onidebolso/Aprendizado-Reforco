import unittest
import tempfile
from pathlib import Path

import numpy as np

from galaxian_neuroevolution import (
    Agente,
    carregar_agente,
    criar_populacao,
    avaliar,
    mutar,
    nova_geracao,
    selecionar,
)


class AmbienteSimulado:
    def __init__(self):
        self.action_space = type("Espaco", (), {"n": 3})()
        self.passos = 0

    def reset(self, seed=None):
        self.passos = 0
        return np.zeros(128, dtype=np.uint8), {}

    def step(self, acao):
        self.passos += 1
        terminou = self.passos == 2
        return np.zeros(128, dtype=np.uint8), 1.5, terminou, False, {}


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
