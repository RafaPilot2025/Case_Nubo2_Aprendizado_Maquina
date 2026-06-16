# -*- coding: utf-8 -*-
"""
Etapa 4 — Comparação das Abordagens.
Consolida as predições de validação dos 7 sistemas (Etapas 2 e 3) e produz:
  1. tabela comparativa completa (métricas + tempo + custo + complexidade);
  2. P@1 por TIPO de query (maiúsculas, embalagem, volume, curta, longa);
  3. análise de sobreposição de erros entre métodos;
  4. gráfico de barras de P@1.
"""
import re
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# caminho relativo ao próprio script: portátil a renomeações/moves da pasta raiz
BASE = Path(__file__).parents[1]
OUT = Path(__file__).parent / "resultados"
OUT.mkdir(parents=True, exist_ok=True)

# (rótulo, arquivo de predições, tempo p/ 250 queries, custo, complexidade)
SISTEMAS = [
    ("TF-IDF char",      BASE / "Etapa02_NLP_Classico/resultados/top5_tfidf_val.csv",
     "1,3 s", "Gratuito", "Baixa"),
    ("BM25",             BASE / "Etapa02_NLP_Classico/resultados/top5_bm25_val.csv",
     "4,6 s", "Gratuito", "Baixa"),
    ("Embeddings E5",    BASE / "Etapa03_Deep_Learning/resultados/top5_embeddings_val.csv",
     "57 s", "Gratuito (CPU local)", "Média"),
    ("Gemini zero-shot", BASE / "Etapa03_Deep_Learning/resultados/top5_gemini_zeroshot_val.csv",
     "1,8 min", "Gratuito (20 req/dia)", "Alta"),
    ("Gemini few-shot",  BASE / "Etapa03_Deep_Learning/resultados/top5_gemini_fewshot_val.csv",
     "1,9 min", "Gratuito (20 req/dia)", "Alta"),
    ("Claude zero-shot", BASE / "Etapa03_Deep_Learning/resultados/top5_claude_zeroshot_val.csv",
     "18,5 min", "~US$ 0,25 / 250 q", "Alta"),
    ("Claude few-shot",  BASE / "Etapa03_Deep_Learning/resultados/top5_claude_fewshot_val.csv",
     "13,6 min", "~US$ 0,30 / 250 q", "Alta"),
]

preds = {}
for rotulo, arq, *_ in SISTEMAS:
    df = pd.read_csv(arq, dtype=str)
    df["rank_correto"] = df["rank_correto"].astype(int)
    preds[rotulo] = df

# ------------------------------------------------- 1. tabela comparativa
linhas = []
for rotulo, _, tempo, custo, compl in SISTEMAS:
    df = preds[rotulo]
    n = len(df)
    p1 = (df["rank_correto"] == 1).mean()
    mrr = (1 / df.loc[df["rank_correto"] > 0, "rank_correto"]).sum() / n
    r5 = (df["rank_correto"] > 0).mean()
    linhas.append({"Abordagem": rotulo, "P@1": round(p1, 4), "MRR@5": round(mrr, 4),
                   "R@5": round(r5, 4), "Tempo (250 q)": tempo, "Custo": custo,
                   "Complexidade": compl})
tabela = pd.DataFrame(linhas).sort_values("P@1", ascending=False)
tabela.to_csv(OUT / "tabela_comparativa.csv", index=False, encoding="utf-8-sig")

# ------------------------------------------------- 2. P@1 por tipo de query
def classificar(q: str) -> list[str]:
    tipos = []
    if q.upper() == q:
        tipos.append("maiúsculas (perfil distribuidor)")
    if re.search(r"C/\s?\d+|\b(?:UND|UNID|UN)\b|\d+\s?X\s?\d+|\bCX\b|\bFD\b|\bPCT\b", q, re.I):
        tipos.append("com termos de embalagem")
    if re.search(r"\d+\s?(?:ml|l|litros?|g|gr|kg)\b", q, re.I):
        tipos.append("com volume/peso")
    n_tok = len(q.split())
    if n_tok <= 4:
        tipos.append("curta (≤4 palavras)")
    if n_tok >= 10:
        tipos.append("longa (≥10 palavras)")
    if not tipos:
        tipos.append("comum")
    return tipos

queries = preds["TF-IDF char"]["query"].tolist()
tipos_por_query = [classificar(q) for q in queries]
todos_tipos = sorted({t for ts in tipos_por_query for t in ts})

seg_linhas = []
for tipo in todos_tipos:
    idx = [i for i, ts in enumerate(tipos_por_query) if tipo in ts]
    linha = {"tipo de query": tipo, "n": len(idx)}
    for rotulo in preds:
        acertos = (preds[rotulo]["rank_correto"].iloc[idx] == 1).mean()
        linha[rotulo] = round(acertos, 3)
    seg_linhas.append(linha)
segmentos = pd.DataFrame(seg_linhas)
segmentos.to_csv(OUT / "p1_por_tipo_de_query.csv", index=False, encoding="utf-8-sig")

# ------------------------------------------------- 3. sobreposição de erros
with open(OUT / "sobreposicao_erros.txt", "w", encoding="utf-8") as f:
    erros = {r: set(df.index[df["rank_correto"] != 1]) for r, df in preds.items()}
    f.write("ERROS DE P@1 POR SISTEMA (validação, 250 queries)\n\n")
    for r, e in erros.items():
        f.write(f"{r}: {len(e)} erros\n")
    todas = set.union(*erros.values())
    todos_erram = set.intersection(*erros.values())
    f.write(f"\nqueries erradas por PELO MENOS um sistema: {len(todas)}\n")
    f.write(f"queries erradas por TODOS os sistemas: {len(todos_erram)}\n")
    for i in sorted(todos_erram):
        f.write(f"   '{queries[i]}'\n      correto: {preds['TF-IDF char']['gold_name'].iloc[i]}\n")
    f.write("\nERROS EXCLUSIVOS (só aquele sistema errou):\n")
    for r, e in erros.items():
        outros = set.union(*(v for k, v in erros.items() if k != r))
        exclusivos = e - outros
        f.write(f"\n{r}: {len(exclusivos)} exclusivos\n")
        for i in sorted(exclusivos):
            f.write(f"   '{queries[i]}'\n")
    # melhor sistema "oráculo" (se pudéssemos escolher o melhor por query)
    oraculo = (250 - len(todos_erram)) / 250
    f.write(f"\nP@1 de um 'oráculo' que escolhesse o melhor sistema por query: {oraculo:.3f}\n")

# ------------------------------------------------- 4. gráfico
fig, ax = plt.subplots(figsize=(9, 4.5))
t = tabela.sort_values("P@1")
cores = ["#4C72B0" if all(s not in a for s in ("Gemini", "E5", "Claude")) else "#DD8452"
         for a in t["Abordagem"]]
ax.barh(t["Abordagem"], t["P@1"] * 100, color=cores)
for i, (v, m) in enumerate(zip(t["P@1"], t["MRR@5"])):
    ax.text(v * 100 + 0.1, i, f"P@1 {v:.1%}  ·  MRR {m:.3f}", va="center", fontsize=9)
ax.set_xlim(90, 100.8)
ax.set_xlabel("P@1 na validação (%)  —  azul: clássico · laranja: deep learning")
ax.set_title("Comparação das abordagens (250 queries de validação)")
plt.tight_layout()
plt.savefig(OUT / "comparacao_p1.png", dpi=130)
plt.close()

print(tabela.to_string(index=False))
print("\nOK - resultados em", OUT)
