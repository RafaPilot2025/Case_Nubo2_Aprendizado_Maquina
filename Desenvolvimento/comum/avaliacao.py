# -*- coding: utf-8 -*-
"""
Script de avaliação — métricas definidas no enunciado (K = 5):
  P@1   : fração de queries com o produto correto em 1º lugar
  MRR@5 : média de 1/posição do produto correto (0 se fora do top-5)
  R@5   : fração de queries com o produto correto em qualquer posição do top-5
"""


def avaliar(top5_ids: list[list[str]], gold_ids: list[str]) -> dict:
    n = len(gold_ids)
    assert n == len(top5_ids) and n > 0
    p1, rr, hits = 0, 0.0, 0
    for ranking, correto in zip(top5_ids, gold_ids):
        if ranking and ranking[0] == correto:
            p1 += 1
        if correto in ranking[:5]:
            rr += 1.0 / (ranking.index(correto) + 1)
            hits += 1
    return {"P@1": p1 / n, "MRR@5": rr / n, "R@5": hits / n}
