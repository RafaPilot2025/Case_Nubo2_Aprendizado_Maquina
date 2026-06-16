# -*- coding: utf-8 -*-
"""
Etapa 6 — Análise Qualitativa.
Produz os 4 blocos exigidos pelo enunciado:
  A) exemplos de ACERTOS (incluindo queries do perfil "distribuidor");
  B) ERROS consolidados das duas abordagens finais (validação + teste);
  C) casos AMBÍGUOS do catálogo (duplicados e quase-duplicados);
  D) experimento NO_MATCH: queries de queries.csv cujo produto não existe
     no catálogo — limiar de similaridade no clássico e abstenção no LLM
     (com grupo de controle para medir falso alarme).
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

sys.path.append(str(Path(__file__).parents[1] / "comum"))
from preprocessamento import normalizar

DADOS = Path(__file__).parents[2] / "Dados_a_trabalhar"
BASE = Path(__file__).parents[1]
OUT = Path(__file__).parent / "resultados"
OUT.mkdir(parents=True, exist_ok=True)

catalog = pd.read_csv(DADOS / "catalog.csv", dtype=str)
queries_dev = pd.read_csv(DADOS / "queries.csv", dtype=str)
val = pd.read_csv(DADOS / "queries_val.csv", dtype=str)

tfidf_val = pd.read_csv(BASE / "Etapa02_NLP_Classico/resultados/top5_tfidf_val.csv", dtype=str)
tfidf_test = pd.read_csv(BASE / "Etapa05_Avaliacao_Final/resultados/top5_tfidf_test.csv", dtype=str)
claude_val = pd.read_csv(BASE / "Etapa03_Deep_Learning/resultados/top5_claude_fewshot_val.csv", dtype=str)
claude_test = pd.read_csv(BASE / "Etapa05_Avaliacao_Final/resultados/top5_claude_fewshot_test.csv", dtype=str)

rel = open(OUT / "analise_qualitativa.txt", "w", encoding="utf-8")
def w(*a): print(*a, file=rel)

# ============================================================ A) ACERTOS
w("=" * 72)
w("A) EXEMPLOS DE ACERTOS")
w("=" * 72)
w("\n--- A1. Queries do perfil 'distribuidor' (maiúsculas/abreviações) acertadas")
w("--- pelo TF-IDF char em 1º lugar:")
dificil = tfidf_val[(tfidf_val["rank_correto"] == "1") &
                    (tfidf_val["query"].str.contains(r"C/\s?\d|UND|UNID|^\S+(?:\s\S+)*$", regex=True)) &
                    (tfidf_val["query"].str.upper() == tfidf_val["query"])]
for _, r in dificil.head(6).iterrows():
    w(f"  '{r['query']}'")
    w(f"     -> {r['gold_name']}  (score {r['pred1_score']})")

w("\n--- A2. Erros do TF-IDF que o Claude CORRIGIU (validação):")
err_tfidf = set(tfidf_val[tfidf_val["rank_correto"] != "1"]["query"])
acertou_claude = claude_val[(claude_val["query"].isin(err_tfidf)) &
                            (claude_val["rank_correto"] == "1")]
for _, r in acertou_claude.iterrows():
    w(f"  '{r['query']}'")
    w(f"     correto : {r['gold_name']}")
    t = tfidf_val[tfidf_val["query"] == r["query"]].iloc[0]
    w(f"     TF-IDF havia escolhido: {t['pred1_name']}")

# ============================================================ B) ERROS
w("\n" + "=" * 72)
w("B) ERROS CONSOLIDADOS DAS ABORDAGENS FINAIS")
w("=" * 72)
for nome, df, conjunto in [("TF-IDF char", tfidf_val, "validação"),
                           ("TF-IDF char", tfidf_test, "teste"),
                           ("Claude few-shot", claude_val, "validação"),
                           ("Claude few-shot", claude_test, "teste")]:
    erros = df[df["rank_correto"] != "1"]
    w(f"\n--- {nome} | {conjunto}: {len(erros)} erros")
    for _, r in erros.iterrows():
        w(f"  query  : {r['query']}")
        w(f"  correto: {r['gold_name']}  (ficou em rank {r['rank_correto']})")
        w(f"  predito: {r['pred1_name']}")

# ============================================================ C) AMBÍGUOS
w("\n" + "=" * 72)
w("C) CASOS AMBÍGUOS — produtos duplicados/quase-duplicados no catálogo")
w("=" * 72)
casos = [("gastronomique", "Manteigas Président (ordem de palavras trocada)"),
         ("sucralose linea", "Adoçante Linea (nomes idênticos, IDs diferentes)"),
         ("sapo de otro pozo", "Vinhos Sapo de Otro Pozo (malbec × blend)"),
         ("cordero con piel", "Vinhos Cordero con Piel de Lobo (malbec × tinto de tintas)")]
for chave, titulo in casos:
    w(f"\n--- {titulo}:")
    achados = catalog[catalog["product_name"].str.lower().str.contains(chave)]
    for _, r in achados.iterrows():
        w(f"  [{r['product_id']}] {r['product_name']}")

# ============================================================ D) NO_MATCH
w("\n" + "=" * 72)
w("D) EXPERIMENTO NO_MATCH")
w("=" * 72)

# vocabulário do catálogo (nomes + marcas)
def tokens(txt):
    return re.findall(r"[a-zà-ÿ]+", str(txt).lower())

vocab = set()
for t in catalog["product_name"]:
    vocab.update(tokens(t))
for t in catalog["brand_name"]:
    vocab.update(tokens(t))

# candidatas a NO_MATCH: query tem token "de marca" (>=5 letras) inexistente no catálogo
def fora_vocab(q):
    ts = [t for t in tokens(q) if len(t) >= 5]
    fora = [t for t in ts if t not in vocab]
    return fora

cand = []
for q in queries_dev["text"].drop_duplicates():
    fora = fora_vocab(q)
    if len(fora) >= 1 and len(tokens(q)) >= 3:
        cand.append((q, fora))
w(f"\nqueries de queries.csv com termo de marca fora do vocabulário do catálogo: {len(cand)}")
rng = np.random.RandomState(42)
idx_sel = rng.choice(len(cand), size=25, replace=False)
no_match = [cand[i] for i in sorted(idx_sel)]
w("amostra analisada (25):")
for q, fora in no_match:
    w(f"  '{q}'  (termos inexistentes no catálogo: {fora})")

# ----- D1: comportamento do clássico — distribuição de scores
docs = [normalizar(n) for n in catalog["product_name"]]
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
matriz = vec.fit_transform(docs)
ids_array = catalog["product_id"].to_numpy()
nome_por_id = dict(zip(catalog["product_id"], catalog["product_name"]))
marca_por_id = dict(zip(catalog["product_id"], catalog["brand_name"]))

def top10(textos):
    sims = linear_kernel(vec.transform([normalizar(t) for t in textos]), matriz)
    return np.argsort(-sims, axis=1)[:, :10], np.sort(sims, axis=1)[:, ::-1][:, :10]

nm_textos = [q for q, _ in no_match]
nm_idx, nm_scores = top10(nm_textos)
val_idx, val_scores = top10(val["text"].tolist())

w("\n--- D1. Abordagem clássica (TF-IDF): score do 1º colocado")
w(f"  queries COM match (validação, n=250): mediana={np.median(val_scores[:,0]):.3f} "
  f"p5={np.percentile(val_scores[:,0],5):.3f} min={val_scores[:,0].min():.3f}")
w(f"  candidatas NO_MATCH (n=25)          : mediana={np.median(nm_scores[:,0]):.3f} "
  f"max={nm_scores[:,0].max():.3f}")
limiar = np.percentile(val_scores[:, 0], 5)
detectadas = (nm_scores[:, 0] < limiar).sum()
falso_alarme = (val_scores[:, 0] < limiar).mean()
w(f"  limiar no percentil 5 da validação ({limiar:.3f}):")
w(f"    NO_MATCH detectadas (score abaixo do limiar): {detectadas}/25")
w(f"    falso alarme na validação: {falso_alarme:.1%}")
w("  exemplos do que o clássico retorna para NO_MATCH (sempre retorna algo!):")
for k in range(4):
    w(f"    '{nm_textos[k]}' -> {nome_por_id[ids_array[nm_idx[k][0]]]} (score {nm_scores[k][0]:.3f})")

# ----- D2: comportamento do LLM — prompt com opção de abstenção
SISTEMA_ABST = """Você é um sistema de matching de produtos de um distribuidor brasileiro.
Dado um pedido de compra escrito livremente, ranqueie os candidatos do catálogo do mais para o menos provável de corresponder ao pedido.
Regras: quantidades de caixa/fardo (C/6, C/12, UND) NÃO fazem parte do produto; volume, peso, sabor e versão (zero, light, sem sal) SÃO decisivos.
IMPORTANTE: se NENHUM candidato for o produto pedido (marca/produto não está na lista), responda {"ranking": []}.
Caso contrário responda {"ranking": [n1, n2, n3, n4, n5]}."""

SCHEMA = {"type": "object",
          "properties": {"ranking": {"type": "array", "items": {"type": "integer"}}},
          "required": ["ranking"], "additionalProperties": False}

client = anthropic.Anthropic(max_retries=5)

def perguntar(query, idx_cands):
    linhas = [f"{n}. {nome_por_id[ids_array[i]]} (marca: {marca_por_id[ids_array[i]]})"
              for n, i in enumerate(idx_cands, start=1)]
    prompt = f'Pedido: "{query}"\nCandidatos:\n' + "\n".join(linhas)
    resp = client.messages.create(
        model="claude-haiku-4-5", max_tokens=256, system=SISTEMA_ABST,
        messages=[{"role": "user", "content": prompt}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}})
    texto = next(b.text for b in resp.content if b.type == "text")
    return json.loads(texto)["ranking"]

w("\n--- D2. LLM (Claude) com instrução de abstenção")
# grupo NO_MATCH
abst_nm, escolhas_nm = 0, []
for k, (q, _) in enumerate(no_match):
    r = perguntar(q, nm_idx[k])
    if not r:
        abst_nm += 1
    else:
        escolhas_nm.append((q, nome_por_id[ids_array[nm_idx[k][r[0]-1]]]))
    time.sleep(1.0)
# grupo controle: 25 queries da validação (que TÊM match)
ctrl_rows = val.sample(25, random_state=42).reset_index()
abst_ctrl, acertos_ctrl = 0, 0
for _, row in ctrl_rows.iterrows():
    k = row["index"]
    r = perguntar(row["text"], val_idx[k])
    if not r:
        abst_ctrl += 1
    elif ids_array[val_idx[k][r[0]-1]] == row["matched_id"]:
        acertos_ctrl += 1
    time.sleep(1.0)

w(f"  NO_MATCH (n=25): abstenções corretas = {abst_nm}/25 ({abst_nm/25:.0%})")
if escolhas_nm:
    w("   casos em que mesmo assim escolheu um produto:")
    for q, p in escolhas_nm[:6]:
        w(f"    '{q}' -> {p}")
w(f"  CONTROLE com match (n=25): abstenções indevidas = {abst_ctrl}/25 ({abst_ctrl/25:.0%}) "
  f"| acertos em 1º = {acertos_ctrl}/{25 - abst_ctrl}")

rel.close()
print("OK - análise em", OUT / "analise_qualitativa.txt")
