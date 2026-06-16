# -*- coding: utf-8 -*-
"""
Etapa 5 — Avaliação Final sobre queries_test.csv (PIPELINES LOCAIS).
Roda as configurações CONGELADAS escolhidas na validação (nenhum ajuste aqui):
  - TF-IDF char_wb (3,5), documento = nome             [campeão Abordagem 1]
  - BM25, documento = nome+marca                       [campeão BM25]
  - Embeddings E5-small, texto pré-processado          [campeão neural local]
O Gemini (few-shot) será avaliado em execução separada, por restrição de cota.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

sys.path.append(str(Path(__file__).parents[1] / "comum"))
from avaliacao import avaliar
from preprocessamento import montar_documento, normalizar

DADOS = Path(__file__).parents[2] / "Dados_a_trabalhar"
OUT = Path(__file__).parent / "resultados"
OUT.mkdir(parents=True, exist_ok=True)

catalog = pd.read_csv(DADOS / "catalog.csv", dtype=str)
test = pd.read_csv(DADOS / "queries_test.csv", dtype=str)
gold = test["matched_id"].tolist()
ids_array = catalog["product_id"].to_numpy()
nome_por_id = dict(zip(catalog["product_id"], catalog["product_name"]))
queries_norm = [normalizar(t) for t in test["text"]]


def top5(scores):
    idx = np.argpartition(-scores, 5)[:5]
    idx = idx[np.argsort(-scores[idx])]
    return ids_array[idx].tolist(), scores[idx].tolist()


def salvar(nome_arq, top_ids, top_scores):
    linhas = []
    for q, g, ids5, sc5 in zip(test["text"], gold, top_ids, top_scores):
        rank = ids5.index(g) + 1 if g in ids5 else 0
        linha = {"query": q, "gold_id": g, "gold_name": nome_por_id[g], "rank_correto": rank}
        for k in range(5):
            linha[f"pred{k+1}_id"] = ids5[k]
            linha[f"pred{k+1}_name"] = nome_por_id[ids5[k]]
            linha[f"pred{k+1}_score"] = round(float(sc5[k]), 4)
        linhas.append(linha)
    pd.DataFrame(linhas).to_csv(OUT / nome_arq, index=False, encoding="utf-8-sig")


resultados = []

# ---------------- 1. TF-IDF char (3,5), nome — campeão da Abordagem 1
docs_nome = [normalizar(n) for n in catalog["product_name"]]
t0 = time.perf_counter()
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
matriz = vec.fit_transform(docs_nome)
sims = linear_kernel(vec.transform(queries_norm), matriz)
ti, ts = zip(*[top5(linha) for linha in sims])
t_tfidf = time.perf_counter() - t0
m = avaliar(list(ti), gold)
resultados.append(("TF-IDF char (3,5) / nome", m, f"{t_tfidf:.1f} s"))
salvar("top5_tfidf_test.csv", ti, ts)

# ---------------- 2. BM25, nome+marca — campeão BM25
docs_marca = [montar_documento(n, b, usar_marca=True)
              for n, b in zip(catalog["product_name"], catalog["brand_name"])]
t0 = time.perf_counter()
bm25 = BM25Okapi([d.split() for d in docs_marca])
ti, ts = [], []
for q in queries_norm:
    a, b = top5(bm25.get_scores(q.split()))
    ti.append(a)
    ts.append(b)
t_bm25 = time.perf_counter() - t0
m = avaliar(ti, gold)
resultados.append(("BM25 / nome+marca", m, f"{t_bm25:.1f} s"))
salvar("top5_bm25_test.csv", ti, ts)

# ---------------- 3. Embeddings E5-small, pré-processado — campeão neural local
modelo = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
t0 = time.perf_counter()
emb_docs = modelo.encode([f"passage: {d}" for d in docs_nome], batch_size=128,
                         normalize_embeddings=True, show_progress_bar=False)
emb_q = modelo.encode([f"query: {q}" for q in queries_norm], batch_size=128,
                      normalize_embeddings=True, show_progress_bar=False)
sims = emb_q @ emb_docs.T
ti, ts = zip(*[top5(linha) for linha in sims])
t_e5 = time.perf_counter() - t0
m = avaliar(list(ti), gold)
resultados.append(("Embeddings E5-small / pré-proc", m, f"{t_e5:.1f} s"))
salvar("top5_embeddings_test.csv", ti, ts)

# ---------------- resumo
with open(OUT / "metricas_teste_locais.txt", "w", encoding="utf-8") as f:
    f.write("AVALIAÇÃO FINAL — queries_test.csv (250 queries) — pipelines locais congelados\n\n")
    f.write(f"{'sistema':35s} {'P@1':>7s} {'MRR@5':>8s} {'R@5':>7s} {'tempo':>10s}\n")
    for nome, m, tempo in resultados:
        f.write(f"{nome:35s} {m['P@1']:7.3f} {m['MRR@5']:8.4f} {m['R@5']:7.3f} {tempo:>10s}\n")
for nome, m, tempo in resultados:
    print(f"{nome:35s} P@1={m['P@1']:.3f} MRR@5={m['MRR@5']:.4f} R@5={m['R@5']:.3f} ({tempo})")
print("OK")
