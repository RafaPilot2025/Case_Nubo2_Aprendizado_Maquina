# -*- coding: utf-8 -*-
"""
Etapa 2 — Abordagem 1: NLP Clássico (TF-IDF × BM25)
Para cada query: pré-processa, calcula similaridade com todos os produtos do
catálogo, retorna top-5 com pontuações e registra o 1º como predição.
Variações testadas na validação (queries_val.csv) para escolher a melhor
configuração de cada método; o teste fica intocado até a Etapa 5.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

sys.path.append(str(Path(__file__).parents[1] / "comum"))
from avaliacao import avaliar
from preprocessamento import montar_documento, normalizar

DADOS = Path(__file__).parents[2] / "Dados_a_trabalhar"
OUT = Path(__file__).parent / "resultados"
OUT.mkdir(exist_ok=True)

catalog = pd.read_csv(DADOS / "catalog.csv", dtype=str)
val = pd.read_csv(DADOS / "queries_val.csv", dtype=str)

queries_norm = [normalizar(t) for t in val["text"]]
gold = val["matched_id"].tolist()
nome_por_id = dict(zip(catalog["product_id"], catalog["product_name"]))


def top5_de_scores(scores: np.ndarray, ids: np.ndarray):
    idx = np.argpartition(-scores, 5)[:5]
    idx = idx[np.argsort(-scores[idx])]
    return ids[idx].tolist(), scores[idx].tolist()


def rodar_tfidf(docs, ids, analyzer, ngram):
    t0 = time.perf_counter()
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram,
                          token_pattern=r"\S+" if analyzer == "word" else None)
    matriz = vec.fit_transform(docs)
    t_index = time.perf_counter() - t0
    t0 = time.perf_counter()
    sims = linear_kernel(vec.transform(queries_norm), matriz)  # cosseno (vetores L2)
    top_ids, top_scores = [], []
    for linha in sims:
        ti, ts = top5_de_scores(linha, ids)
        top_ids.append(ti)
        top_scores.append(ts)
    t_query = time.perf_counter() - t0
    return top_ids, top_scores, t_index, t_query


def rodar_bm25(docs, ids):
    t0 = time.perf_counter()
    bm25 = BM25Okapi([d.split() for d in docs])
    t_index = time.perf_counter() - t0
    t0 = time.perf_counter()
    top_ids, top_scores = [], []
    for q in queries_norm:
        scores = bm25.get_scores(q.split())
        ti, ts = top5_de_scores(scores, ids)
        top_ids.append(ti)
        top_scores.append(ts)
    t_query = time.perf_counter() - t0
    return top_ids, top_scores, t_index, t_query


def salvar_predicoes(nome_arq, top_ids, top_scores):
    linhas = []
    for q_orig, q_norm, g, ids5, sc5 in zip(val["text"], queries_norm, gold, top_ids, top_scores):
        rank = ids5.index(g) + 1 if g in ids5 else 0  # 0 = fora do top-5
        linha = {"query": q_orig, "query_norm": q_norm, "gold_id": g,
                 "gold_name": nome_por_id[g], "rank_correto": rank}
        for k in range(5):
            linha[f"pred{k+1}_id"] = ids5[k]
            linha[f"pred{k+1}_name"] = nome_por_id[ids5[k]]
            linha[f"pred{k+1}_score"] = round(sc5[k], 4)
        linhas.append(linha)
    pd.DataFrame(linhas).to_csv(OUT / nome_arq, index=False, encoding="utf-8-sig")


ids_array = catalog["product_id"].to_numpy()
resultados = []
predicoes_campeas = {}

for usar_marca in (False, True):
    docs = [montar_documento(n, m, usar_marca)
            for n, m in zip(catalog["product_name"], catalog["brand_name"])]
    rotulo_doc = "nome+marca" if usar_marca else "nome"

    for analyzer, ngram, rotulo in [("word", (1, 1), "TF-IDF palavra (1,1)"),
                                    ("word", (1, 2), "TF-IDF palavra (1,2)"),
                                    ("char_wb", (3, 5), "TF-IDF char (3,5)")]:
        ti, ts, t_idx, t_qry = rodar_tfidf(docs, ids_array, analyzer, ngram)
        m = avaliar(ti, gold)
        resultados.append({"método": rotulo, "documento": rotulo_doc, **m,
                           "t_indexação(s)": round(t_idx, 2), "t_250queries(s)": round(t_qry, 2)})
        predicoes_campeas[(rotulo, rotulo_doc)] = (ti, ts)

    ti, ts, t_idx, t_qry = rodar_bm25(docs, ids_array)
    m = avaliar(ti, gold)
    resultados.append({"método": "BM25", "documento": rotulo_doc, **m,
                       "t_indexação(s)": round(t_idx, 2), "t_250queries(s)": round(t_qry, 2)})
    predicoes_campeas[("BM25", rotulo_doc)] = (ti, ts)

df_res = pd.DataFrame(resultados).sort_values("P@1", ascending=False)
df_res.to_csv(OUT / "metricas_validacao.csv", index=False, encoding="utf-8-sig")

# salva predições top-5 da melhor variação de cada família (p/ análise qualitativa)
melhor_tfidf = df_res[df_res["método"].str.startswith("TF-IDF")].iloc[0]
melhor_bm25 = df_res[df_res["método"] == "BM25"].iloc[0]
for melhor, arq in [(melhor_tfidf, "top5_tfidf_val.csv"), (melhor_bm25, "top5_bm25_val.csv")]:
    ti, ts = predicoes_campeas[(melhor["método"], melhor["documento"])]
    salvar_predicoes(arq, ti, ts)

with open(OUT / "resumo.txt", "w", encoding="utf-8") as f:
    f.write("RESULTADOS NA VALIDAÇÃO (250 queries, catálogo de 14.206 produtos)\n\n")
    f.write(df_res.to_string(index=False))
    f.write("\n\nMelhor TF-IDF: " + f"{melhor_tfidf['método']} / {melhor_tfidf['documento']}\n")
    f.write("Melhor BM25  : " + f"BM25 / {melhor_bm25['documento']}\n")
print("OK - resultados em", OUT)
