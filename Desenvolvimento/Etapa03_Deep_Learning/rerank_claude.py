# -*- coding: utf-8 -*-
"""
Etapa 3/5 — Abordagem 2 (Deep Learning), Estratégia B: LLM como reranker.
Versão Claude (Anthropic) — substitui o Gemini após a inviabilidade da cota
gratuita (20 req/dia/modelo) e os 503 crônicos documentados na Etapa 3.

Pipeline híbrido (recomendação do professor):
  1. TF-IDF char(3,5) (campeão da Etapa 2) filtra os top-10 candidatos;
  2. o Claude (claude-haiku-4-5) ranqueia os 5 melhores.

Modelo: claude-haiku-4-5 — decisão de custo aprovada pelo grupo
(~US$ 0,21 por rodada de 250 queries; pago com créditos existentes).
Uma chamada por query (sem lotes — não há cota diária na API paga),
com saída estruturada (JSON Schema) e checkpoint retomável por query.
A API key é lida da variável de ambiente ANTHROPIC_API_KEY.

Uso:  python rerank_claude.py [zero|few] [val|test]
"""
import json
import sys
import time
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

sys.path.append(str(Path(__file__).parents[1] / "comum"))
from avaliacao import avaliar
from preprocessamento import normalizar

MODO = (sys.argv[1] if len(sys.argv) > 1 else "zero").lower()
SPLIT = (sys.argv[2] if len(sys.argv) > 2 else "val").lower()
assert MODO in ("zero", "few") and SPLIT in ("val", "test")

MODELO = "claude-haiku-4-5"
INTERVALO_S = 1.3          # ~46 req/min, abaixo do limite de 50 RPM do tier 1
N_CANDIDATOS = 10

DADOS = Path(__file__).parents[2] / "Dados_a_trabalhar"
BASE = Path(__file__).parents[1]
OUT = (BASE / "Etapa03_Deep_Learning" / "resultados" if SPLIT == "val"
       else BASE / "Etapa05_Avaliacao_Final" / "resultados")
OUT.mkdir(parents=True, exist_ok=True)

client = anthropic.Anthropic(max_retries=5)  # lê ANTHROPIC_API_KEY do ambiente

catalog = pd.read_csv(DADOS / "catalog.csv", dtype=str)
arquivo_queries = "queries_val.csv" if SPLIT == "val" else "queries_test.csv"
df_q = pd.read_csv(DADOS / arquivo_queries, dtype=str)
gold = df_q["matched_id"].tolist()
ids_array = catalog["product_id"].to_numpy()
nome_por_id = dict(zip(catalog["product_id"], catalog["product_name"]))
marca_por_id = dict(zip(catalog["product_id"], catalog["brand_name"]))

# ----- passo 1: filtro top-10 com o campeão da Etapa 2 (TF-IDF char 3-5, só nome)
docs = [normalizar(n) for n in catalog["product_name"]]
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
matriz = vec.fit_transform(docs)
sims = linear_kernel(vec.transform([normalizar(t) for t in df_q["text"]]), matriz)
top10_idx = np.argsort(-sims, axis=1)[:, :N_CANDIDATOS]

FEW_SHOT = """
Exemplos de como resolver a tarefa:

Pedido: "COCA COLA 1L C/6"
Candidatos: 1. Refrigerante coca-cola pet 1 litro | 2. Refrigerante coca-cola 2 litros | 3. Refrigerante coca-cola lata 350ml
Resposta: {"ranking": [1, 2, 3]}
(O pedido cita 1L; "C/6" é só a quantidade da caixa e deve ser ignorada.)

Pedido: "SKOL LATA 350ml C/12"
Candidatos: 1. Cerveja skol garrafa 600ml | 2. Cerveja skol lata 350ml | 3. Cerveja skol pilsen 269ml
Resposta: {"ranking": [2, 3, 1]}
(Volume e embalagem "lata" decidem.)

Pedido: "MONSTER BRANCO S/ACUCAR C/6 UNID"
Candidatos: 1. Energético monster energy 473ml | 2. Energético monster energy zero açúcar lata 473ml | 3. Refrigerante branco soda 2 litros
Resposta: {"ranking": [2, 1, 3]}
("BRANCO" é a cor da lata da versão zero açúcar; "S/ACUCAR" = sem açúcar.)
""".strip()

SISTEMA = """Você é um sistema de matching de produtos de um distribuidor brasileiro.
Dado um pedido de compra escrito livremente (com abreviações e erros), ranqueie os candidatos do catálogo do mais para o menos provável de corresponder ao pedido.
Regras: quantidades de caixa/fardo (C/6, C/12, UND) NÃO fazem parte do produto; volume, peso, sabor e versão (zero, light, sem sal) SÃO decisivos.
Responda com JSON: {"ranking": [n1, n2, n3, n4, n5]} — os números dos 5 candidatos mais prováveis."""

SCHEMA_RANKING = {
    "type": "object",
    "properties": {
        "ranking": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["ranking"],
    "additionalProperties": False,
}


def montar_prompt(qi: int) -> str:
    linhas = [f"{n}. {nome_por_id[ids_array[i]]} (marca: {marca_por_id[ids_array[i]]})"
              for n, i in enumerate(top10_idx[qi], start=1)]
    exemplos = FEW_SHOT + "\n\n" if MODO == "few" else ""
    return f'{exemplos}Pedido: "{df_q["text"].iloc[qi]}"\nCandidatos:\n' + "\n".join(linhas)


def chamar_claude(prompt: str) -> list[int]:
    resp = client.messages.create(
        model=MODELO,
        max_tokens=256,
        system=SISTEMA,
        messages=[{"role": "user", "content": prompt}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA_RANKING}},
    )
    texto = next(b.text for b in resp.content if b.type == "text")
    ranking = [int(x) for x in json.loads(texto)["ranking"]
               if 1 <= int(x) <= N_CANDIDATOS]
    return list(dict.fromkeys(ranking))[:5]


# checkpoint por query: retomável após qualquer interrupção
CHECKPOINT = OUT / f"checkpoint_claude_{MODO}shot_{SPLIT}.jsonl"
feitos = {}
if CHECKPOINT.exists():
    with open(CHECKPOINT, encoding="utf-8") as f:
        for linha in f:
            reg = json.loads(linha)
            feitos[reg["qi"]] = reg["ranking"]
    print(f"checkpoint: {len(feitos)}/{len(df_q)} queries já processadas")

fallbacks = 0
t0 = time.perf_counter()
with open(CHECKPOINT, "a", encoding="utf-8") as ckpt:
    for qi in range(len(df_q)):
        if qi in feitos:
            continue
        try:
            ranking = chamar_claude(montar_prompt(qi))
            if not ranking:
                raise ValueError("ranking vazio")
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"  query {qi}: fallback (ordem TF-IDF) — {e}")
            ranking = [1, 2, 3, 4, 5]
            fallbacks += 1
        feitos[qi] = ranking
        ckpt.write(json.dumps({"qi": qi, "ranking": ranking}) + "\n")
        ckpt.flush()
        if (qi + 1) % 25 == 0:
            print(f"  {qi+1}/{len(df_q)}")
        time.sleep(INTERVALO_S)
t_total = time.perf_counter() - t0

# monta top-5 final por query (completa com a ordem do TF-IDF se necessário)
top5_ids = []
for qi in range(len(df_q)):
    ids5 = [ids_array[top10_idx[qi][n - 1]] for n in feitos[qi]]
    for i in top10_idx[qi]:
        if len(ids5) >= 5:
            break
        if ids_array[i] not in ids5:
            ids5.append(ids_array[i])
    top5_ids.append(ids5)

m = avaliar(top5_ids, gold)
print(f"\nClaude ({MODELO}) {MODO}-shot [{SPLIT}]: "
      f"P@1={m['P@1']:.3f} MRR@5={m['MRR@5']:.4f} R@5={m['R@5']:.3f} "
      f"| tempo: {t_total/60:.1f} min | fallbacks: {fallbacks}")

linhas = []
for qi, (g, ids5) in enumerate(zip(gold, top5_ids)):
    rank = ids5.index(g) + 1 if g in ids5 else 0
    linha = {"query": df_q["text"].iloc[qi], "gold_id": g, "gold_name": nome_por_id[g],
             "rank_correto": rank}
    for k in range(5):
        linha[f"pred{k+1}_id"] = ids5[k]
        linha[f"pred{k+1}_name"] = nome_por_id[ids5[k]]
    linhas.append(linha)
pd.DataFrame(linhas).to_csv(OUT / f"top5_claude_{MODO}shot_{SPLIT}.csv",
                            index=False, encoding="utf-8-sig")
with open(OUT / f"metricas_claude_{MODO}shot_{SPLIT}.txt", "w", encoding="utf-8") as f:
    f.write(f"Claude ({MODELO}) {MODO}-shot reranking top-{N_CANDIDATOS} "
            f"(filtro TF-IDF char) — conjunto: {SPLIT}\n")
    f.write(f"P@1={m['P@1']:.4f}  MRR@5={m['MRR@5']:.4f}  R@5={m['R@5']:.4f}\n")
    f.write(f"tempo total {len(df_q)} queries: {t_total/60:.1f} min | fallbacks: {fallbacks}\n")
