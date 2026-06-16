# -*- coding: utf-8 -*-
"""
Etapa 3 — Abordagem 2 (Deep Learning), Estratégia B: LLM como reranker.
Pipeline híbrido recomendado pelo professor:
  1. TF-IDF char(3,5) (campeão da Etapa 2) filtra os top-10 candidatos;
  2. o LLM (Google Gemini) lê a query + candidatos e devolve o ranking top-5.

RESTRIÇÃO REAL DE PRODUÇÃO (verificada em 09/06/2026): o nível gratuito da
API permite apenas 20 requisições/dia por modelo. Por isso as queries são
enviadas em LOTES de 32 por chamada (250 queries -> 8 chamadas por modo).

Modos: zero-shot (sem exemplos) e few-shot (3 exemplos no prompt).
A API key é lida da variável de ambiente GEMINI_API_KEY (nunca fica no código).

Uso:  python rerank_gemini.py [zero|few]
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

sys.path.append(str(Path(__file__).parents[1] / "comum"))
from avaliacao import avaliar
from preprocessamento import normalizar

MODO = (sys.argv[1] if len(sys.argv) > 1 else "zero").lower()
assert MODO in ("zero", "few")
# 2.5-flash apresentou 503 crônico no free tier (2 dias seguidos, 15+ tentativas);
# flash-lite tem fila própria menos congestionada e mesma cota (20 req/dia)
MODELO = "gemini-2.5-flash-lite"
TAM_LOTE = 32
INTERVALO_S = 10.0
N_CANDIDATOS = 10

DADOS = Path(__file__).parents[2] / "Dados_a_trabalhar"
OUT = Path(__file__).parent / "resultados"
OUT.mkdir(parents=True, exist_ok=True)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

catalog = pd.read_csv(DADOS / "catalog.csv", dtype=str)
val = pd.read_csv(DADOS / "queries_val.csv", dtype=str)
gold = val["matched_id"].tolist()
ids_array = catalog["product_id"].to_numpy()
nome_por_id = dict(zip(catalog["product_id"], catalog["product_name"]))
marca_por_id = dict(zip(catalog["product_id"], catalog["brand_name"]))

# ----- passo 1: filtro top-10 com o campeão da Etapa 2 (TF-IDF char 3-5, só nome)
docs = [normalizar(n) for n in catalog["product_name"]]
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
matriz = vec.fit_transform(docs)
sims = linear_kernel(vec.transform([normalizar(t) for t in val["text"]]), matriz)
top10_idx = np.argsort(-sims, axis=1)[:, :N_CANDIDATOS]

FEW_SHOT = """
Exemplos de como resolver cada pedido:

Pedido: "COCA COLA 1L C/6"
Candidatos: 1. Refrigerante coca-cola pet 1 litro | 2. Refrigerante coca-cola 2 litros | 3. Refrigerante coca-cola lata 350ml
Ranking correto: [1, 2, 3]
(O pedido cita 1L; "C/6" é só a quantidade da caixa e deve ser ignorada.)

Pedido: "SKOL LATA 350ml C/12"
Candidatos: 1. Cerveja skol garrafa 600ml | 2. Cerveja skol lata 350ml | 3. Cerveja skol pilsen 269ml
Ranking correto: [2, 3, 1]
(Volume e embalagem "lata" decidem.)

Pedido: "MONSTER BRANCO S/ACUCAR C/6 UNID"
Candidatos: 1. Energético monster energy 473ml | 2. Energético monster energy zero açúcar lata 473ml | 3. Refrigerante branco soda 2 litros
Ranking correto: [2, 1, 3]
("BRANCO" é a cor da lata da versão zero açúcar; "S/ACUCAR" = sem açúcar.)
""".strip()

INSTRUCOES = """Você é um sistema de matching de produtos de um distribuidor brasileiro.
Para CADA pedido numerado abaixo, escolha entre os candidatos daquele pedido o produto do catálogo correspondente e ranqueie os 5 mais prováveis (do mais para o menos provável).
Regras: quantidades de caixa/fardo (C/6, C/12, UND) NÃO fazem parte do produto; volume, peso, sabor e versão (zero, light, sem sal) SÃO decisivos."""


def montar_prompt_lote(indices_lote):
    blocos = []
    for qi in indices_lote:
        linhas = [f"  {n}. {nome_por_id[ids_array[i]]} (marca: {marca_por_id[ids_array[i]]})"
                  for n, i in enumerate(top10_idx[qi], start=1)]
        blocos.append(f'Pedido {qi}: "{val["text"].iloc[qi]}"\nCandidatos:\n' + "\n".join(linhas))
    corpo = "\n\n".join(blocos)
    exemplos = "\n\n" + FEW_SHOT if MODO == "few" else ""
    return f"""{INSTRUCOES}{exemplos}

{corpo}

Responda APENAS com JSON, uma linha por pedido, no formato:
{{"resultados": [{{"pedido": <número do pedido>, "ranking": [n1, n2, n3, n4, n5]}}, ...]}}
Inclua TODOS os {len(indices_lote)} pedidos, na mesma ordem."""


def chamar_gemini(prompt: str) -> str:
    # 503 = sobrecarga transitória do servidor (comum no nível gratuito):
    # vale a pena insistir bastante antes de desistir
    for tentativa in range(8):
        try:
            resp = client.models.generate_content(model=MODELO, contents=prompt)
            return resp.text or ""
        except Exception as e:
            espera = min(45 * (tentativa + 1), 240)
            print(f"  erro API ({str(e)[:90]}), aguardando {espera}s...")
            time.sleep(espera)
    raise RuntimeError("API falhou 8 vezes seguidas (checkpoint preservado)")


def extrair_rankings_lote(texto: str, indices_lote) -> dict:
    """Devolve {qi: ranking}; pedidos ausentes ficam fora (fallback depois)."""
    out = {}
    m = re.search(r"\{.*\}", texto, re.DOTALL)
    if m:
        try:
            dados = json.loads(m.group(0))
            for item in dados.get("resultados", []):
                qi = int(item.get("pedido", -1))
                r = [int(x) for x in item.get("ranking", []) if 1 <= int(x) <= N_CANDIDATOS]
                if qi in indices_lote and r:
                    out[qi] = list(dict.fromkeys(r))[:5]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return out


# checkpoint por lote: retomável se a cota diária estourar
CHECKPOINT = OUT / f"checkpoint_gemini_{MODO}shot.jsonl"
feitos = {}
if CHECKPOINT.exists():
    with open(CHECKPOINT, encoding="utf-8") as f:
        for linha in f:
            reg = json.loads(linha)
            feitos.update({int(k): v for k, v in reg["rankings"].items()})
    print(f"checkpoint encontrado: {len(feitos)}/250 queries já processadas")

lotes = [list(range(i, min(i + TAM_LOTE, len(val)))) for i in range(0, len(val), TAM_LOTE)]
respostas_brutas = []
fallbacks = 0
t0 = time.perf_counter()
with open(CHECKPOINT, "a", encoding="utf-8") as ckpt:
    for li, lote in enumerate(lotes):
        pendentes = [qi for qi in lote if qi not in feitos]
        if not pendentes:
            continue
        resposta = chamar_gemini(montar_prompt_lote(pendentes))
        respostas_brutas.append({"lote": li, "resposta": resposta.strip()})
        rankings = extrair_rankings_lote(resposta, pendentes)
        for qi in pendentes:
            if qi not in rankings:      # fallback: mantém a ordem do TF-IDF
                rankings[qi] = [1, 2, 3, 4, 5]
                fallbacks += 1
        feitos.update(rankings)
        ckpt.write(json.dumps({"lote": li, "rankings": {str(k): v for k, v in rankings.items()}},
                              ensure_ascii=False) + "\n")
        ckpt.flush()
        print(f"  lote {li+1}/{len(lotes)} ok ({len(feitos)}/250 queries)")
        time.sleep(INTERVALO_S)
t_total = time.perf_counter() - t0

# monta top-5 final por query
top5_ids = []
for qi in range(len(val)):
    ranking = feitos[qi]
    ids5 = [ids_array[top10_idx[qi][n - 1]] for n in ranking]
    for i in top10_idx[qi]:             # completa até 5 com a ordem do TF-IDF
        if len(ids5) >= 5:
            break
        if ids_array[i] not in ids5:
            ids5.append(ids_array[i])
    top5_ids.append(ids5)

m = avaliar(top5_ids, gold)
print(f"\nGemini {MODO}-shot: P@1={m['P@1']:.3f} MRR@5={m['MRR@5']:.4f} R@5={m['R@5']:.3f} "
      f"| tempo: {t_total/60:.1f} min | fallbacks de parse: {fallbacks}")

if respostas_brutas:
    pd.DataFrame(respostas_brutas).to_json(OUT / f"respostas_gemini_{MODO}shot.jsonl",
                                           orient="records", lines=True, force_ascii=False)
linhas = []
for qi, (g, ids5) in enumerate(zip(gold, top5_ids)):
    rank = ids5.index(g) + 1 if g in ids5 else 0
    linha = {"query": val["text"].iloc[qi], "gold_id": g, "gold_name": nome_por_id[g],
             "rank_correto": rank}
    for k in range(5):
        linha[f"pred{k+1}_id"] = ids5[k]
        linha[f"pred{k+1}_name"] = nome_por_id[ids5[k]]
    linhas.append(linha)
pd.DataFrame(linhas).to_csv(OUT / f"top5_gemini_{MODO}shot_val.csv",
                            index=False, encoding="utf-8-sig")
with open(OUT / f"metricas_gemini_{MODO}shot.txt", "w", encoding="utf-8") as f:
    f.write(f"Gemini ({MODELO}) {MODO}-shot reranking top-{N_CANDIDATOS} "
            f"(filtro TF-IDF char, lotes de {TAM_LOTE})\n")
    f.write(f"P@1={m['P@1']:.4f}  MRR@5={m['MRR@5']:.4f}  R@5={m['R@5']:.4f}\n")
    f.write(f"tempo total 250 queries: {t_total/60:.1f} min | fallbacks de parse: {fallbacks}\n")
