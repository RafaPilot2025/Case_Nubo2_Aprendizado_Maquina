# -*- coding: utf-8 -*-
"""
Etapa 3 — Abordagem 2 (Deep Learning), Estratégia A: Embeddings Semânticos.
Gera vetores densos para queries e produtos com modelos de rede neural
(Sentence-Transformers) e ranqueia por similaridade de cosseno.
Variações testadas na validação:
  - 2 modelos multilíngues (MiniLM paraphrase × E5-small)
  - texto cru × texto pré-processado (mesma normalização da Etapa 2)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).parents[1] / "comum"))
from avaliacao import avaliar
from preprocessamento import normalizar

DADOS = Path(__file__).parents[2] / "Dados_a_trabalhar"
OUT = Path(__file__).parent / "resultados"
OUT.mkdir(parents=True, exist_ok=True)

catalog = pd.read_csv(DADOS / "catalog.csv", dtype=str)
val = pd.read_csv(DADOS / "queries_val.csv", dtype=str)
gold = val["matched_id"].tolist()
ids_array = catalog["product_id"].to_numpy()
nome_por_id = dict(zip(catalog["product_id"], catalog["product_name"]))

MODELOS = {
    "MiniLM-multilingual": ("paraphrase-multilingual-MiniLM-L12-v2", "", ""),
    # E5 exige prefixos distintos para consulta e documento
    "E5-small-multilingual": ("intfloat/multilingual-e5-small", "query: ", "passage: "),
}


def top5(sims_linha):
    idx = np.argpartition(-sims_linha, 5)[:5]
    idx = idx[np.argsort(-sims_linha[idx])]
    return ids_array[idx].tolist(), sims_linha[idx].tolist()


resultados = []
predicoes = {}

for rotulo_modelo, (nome_hf, pref_q, pref_d) in MODELOS.items():
    print(f"carregando {rotulo_modelo}...")
    modelo = SentenceTransformer(nome_hf, device="cpu")
    for usar_preproc in (False, True):
        rotulo_texto = "pré-processado" if usar_preproc else "cru"
        docs = [normalizar(t) if usar_preproc else t for t in catalog["product_name"]]
        qs = [normalizar(t) if usar_preproc else t for t in val["text"]]
        docs = [pref_d + d for d in docs]
        qs = [pref_q + q for q in qs]

        t0 = time.perf_counter()
        emb_docs = modelo.encode(docs, batch_size=128, normalize_embeddings=True,
                                 show_progress_bar=False)
        t_index = time.perf_counter() - t0

        t0 = time.perf_counter()
        emb_q = modelo.encode(qs, batch_size=128, normalize_embeddings=True,
                              show_progress_bar=False)
        sims = emb_q @ emb_docs.T          # cosseno (vetores normalizados)
        top_ids, top_scores = [], []
        for linha in sims:
            ti, ts = top5(linha)
            top_ids.append(ti)
            top_scores.append(ts)
        t_query = time.perf_counter() - t0

        m = avaliar(top_ids, gold)
        resultados.append({"modelo": rotulo_modelo, "texto": rotulo_texto, **m,
                           "t_indexação(s)": round(t_index, 1),
                           "t_250queries(s)": round(t_query, 2)})
        predicoes[(rotulo_modelo, rotulo_texto)] = (top_ids, top_scores)
        print(f"  {rotulo_modelo} / {rotulo_texto}: P@1={m['P@1']:.3f}")

df = pd.DataFrame(resultados).sort_values("P@1", ascending=False)
df.to_csv(OUT / "metricas_validacao_embeddings.csv", index=False, encoding="utf-8-sig")

# predições do campeão (para análise qualitativa)
campeao = df.iloc[0]
top_ids, top_scores = predicoes[(campeao["modelo"], campeao["texto"])]
linhas = []
for q_orig, g, ids5, sc5 in zip(val["text"], gold, top_ids, top_scores):
    rank = ids5.index(g) + 1 if g in ids5 else 0
    linha = {"query": q_orig, "gold_id": g, "gold_name": nome_por_id[g], "rank_correto": rank}
    for k in range(5):
        linha[f"pred{k+1}_id"] = ids5[k]
        linha[f"pred{k+1}_name"] = nome_por_id[ids5[k]]
        linha[f"pred{k+1}_score"] = round(sc5[k], 4)
    linhas.append(linha)
pd.DataFrame(linhas).to_csv(OUT / "top5_embeddings_val.csv", index=False, encoding="utf-8-sig")

with open(OUT / "resumo_embeddings.txt", "w", encoding="utf-8") as f:
    f.write("EMBEDDINGS SEMÂNTICOS — VALIDAÇÃO (250 queries, catálogo 14.206)\n\n")
    f.write(df.to_string(index=False))
    f.write(f"\n\nCampeão: {campeao['modelo']} / texto {campeao['texto']}\n")
print("OK")
