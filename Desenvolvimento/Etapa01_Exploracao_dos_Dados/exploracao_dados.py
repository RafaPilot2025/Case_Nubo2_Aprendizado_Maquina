# -*- coding: utf-8 -*-
"""
Etapa 1 — Exploração dos Dados | T2: Matching de Produtos
Carrega catalog.csv e queries.csv, analisa distribuições, padrões nas
queries (abreviações, embalagens, categorias) e casos desafiadores.
Saídas: relatorio_exploracao.txt + gráficos PNG na pasta resultados/.
"""
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DADOS = Path(__file__).parents[2] / "Dados_a_trabalhar"
OUT = Path(__file__).parent / "resultados"
OUT.mkdir(exist_ok=True)

rel = open(OUT / "relatorio_exploracao.txt", "w", encoding="utf-8")


def w(*args):
    print(*args, file=rel)


# dtype=str preserva zeros à esquerda dos códigos EAN/GTIN (ex.: 07891...)
catalog = pd.read_csv(DADOS / "catalog.csv", dtype=str)
queries = pd.read_csv(DADOS / "queries.csv", dtype=str)
val = pd.read_csv(DADOS / "queries_val.csv", dtype=str)
# queries_test.csv NÃO é carregado aqui: o teste só é aberto na Etapa 5
# (avaliação final), preservando o protocolo validação x teste.

# ---------------------------------------------------------------- 1. visão geral
w("=" * 70)
w("1. VISÃO GERAL")
w("=" * 70)
for nome, df in [("catalog", catalog), ("queries", queries), ("val", val)]:
    w(f"{nome}: {df.shape[0]} linhas x {df.shape[1]} colunas | nulos por coluna: {df.isna().sum().to_dict()}")

w()
w("catalog: product_id únicos:", catalog["product_id"].nunique())
w("catalog: product_name únicos:", catalog["product_name"].nunique())
w("catalog: product_name duplicados (mesmo nome, ids diferentes):",
  catalog.shape[0] - catalog["product_name"].nunique())
dup_names = catalog[catalog.duplicated("product_name", keep=False)].sort_values("product_name")
w("exemplos de nomes duplicados:")
for nome_prod, grupo in list(dup_names.groupby("product_name"))[:5]:
    w(f"   '{nome_prod}' -> ids: {grupo['product_id'].tolist()}")
w()
w("queries: textos únicos:", queries["text"].nunique(), "| duplicadas:",
  queries.shape[0] - queries["text"].nunique())
w("queries duplicadas mais comuns:")
for txt, n in queries["text"].value_counts().head(5).items():
    w(f"   {n}x '{txt}'")

# sobreposição entre conjuntos
w()
w("sobreposição de textos: queries ∩ val:", len(set(queries['text']) & set(val['text'])))

# ------------------------------------------- 2. matched_id de val no catálogo
w()
w("=" * 70)
w("2. ANOTAÇÕES (val) × CATÁLOGO — há NO_MATCH?")
w("=" * 70)
ids_catalogo = set(catalog["product_id"])
for nome, df in [("val", val)]:
    presentes = df["matched_id"].isin(ids_catalogo).sum()
    w(f"{nome}: {presentes}/{df.shape[0]} matched_id existem no catálogo")
    fora = df[~df["matched_id"].isin(ids_catalogo)]
    if not fora.empty:
        w(f"   valores de matched_id FORA do catálogo: {fora['matched_id'].unique().tolist()}")
        w(f"   exemplos de queries: {fora['text'].head(5).tolist()}")

# ---------------------------------------------------------- 3. comprimento dos textos
w()
w("=" * 70)
w("3. COMPRIMENTO DOS TEXTOS")
w("=" * 70)
def stats_len(s):
    chars = s.str.len()
    toks = s.str.split().str.len()
    return (f"chars: média={chars.mean():.1f} mediana={chars.median():.0f} "
            f"min={chars.min()} max={chars.max()} | "
            f"tokens: média={toks.mean():.1f} mediana={toks.median():.0f} "
            f"min={toks.min()} max={toks.max()}")

w("queries.text       :", stats_len(queries["text"]))
w("catalog.product_name:", stats_len(catalog["product_name"]))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
queries["text"].str.split().str.len().hist(bins=range(0, 25), ax=ax[0], color="#4C72B0")
ax[0].set_title("Nº de palavras por query")
ax[0].set_xlabel("palavras")
ax[0].set_ylabel("frequência")
catalog["product_name"].str.split().str.len().hist(bins=range(0, 25), ax=ax[1], color="#DD8452")
ax[1].set_title("Nº de palavras por nome de produto (catálogo)")
ax[1].set_xlabel("palavras")
plt.tight_layout()
plt.savefig(OUT / "dist_comprimento.png", dpi=120)
plt.close()

# ------------------------------------------------------------------- 4. marcas
w()
w("=" * 70)
w("4. MARCAS NO CATÁLOGO")
w("=" * 70)
w("marcas únicas:", catalog["brand_name"].nunique())
top_marcas = catalog["brand_name"].value_counts().head(20)
w("top 20 marcas por nº de produtos:")
for m, n in top_marcas.items():
    w(f"   {n:5d}  {m}")

top_marcas.sort_values().plot(kind="barh", figsize=(8, 6), color="#4C72B0")
plt.title("Top 20 marcas do catálogo (nº de produtos)")
plt.tight_layout()
plt.savefig(OUT / "top_marcas.png", dpi=120)
plt.close()

# ------------------------------------------------------- 5. categorias (1ª palavra)
w()
w("=" * 70)
w("5. CATEGORIAS DE PRODUTO (1ª palavra do product_name como proxy)")
w("=" * 70)
cat_proxy = catalog["product_name"].str.split().str[0].str.lower().value_counts()
w("categorias distintas (proxy):", cat_proxy.shape[0])
w("top 25 categorias:")
for c, n in cat_proxy.head(25).items():
    w(f"   {n:5d}  {c}")

cat_proxy.head(20).sort_values().plot(kind="barh", figsize=(8, 6), color="#55A868")
plt.title("Top 20 categorias (1ª palavra do nome do produto)")
plt.tight_layout()
plt.savefig(OUT / "top_categorias.png", dpi=120)
plt.close()

# --------------------------------------------------------------- 6. caixa e acentos
w()
w("=" * 70)
w("6. CAIXA ALTA E ACENTUAÇÃO")
w("=" * 70)
def tem_acento(s):
    return s != unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()

q = queries["text"]
w(f"queries 100% MAIÚSCULAS: {(q.str.upper() == q).mean():.1%}")
w(f"queries com acento: {q.map(tem_acento).mean():.1%}")
c = catalog["product_name"]
w(f"catálogo 100% MAIÚSCULAS: {(c.str.upper() == c).mean():.1%}")
w(f"catálogo com acento: {c.map(tem_acento).mean():.1%}")

# ----------------------------------------------------- 7. padrões e abreviações
w()
w("=" * 70)
w("7. PADRÕES DE EMBALAGEM/ABREVIAÇÃO NAS QUERIES (regex)")
w("=" * 70)
padroes = {
    "C/N (caixa com N, ex C/6)":      r"\bC/\s?\d+",
    "NxM (ex 12X15)":                 r"\b\d+\s?[xX]\s?\d+",
    "UND/UNID/UN":                    r"\b(?:UND|UNID|UN)\b",
    "CX (caixa)":                     r"\bCX\b",
    "FD/FARDO":                       r"\bFD\b|\bFARDO\b",
    "PCT/PACOTE":                     r"\bPCT\b",
    "volume ml (350ml, 350 ML)":      r"\d+\s?[mM][lL]\b",
    "volume L (1L, 2 L)":             r"\d+\s?[lL]\b",
    "LT/LATA":                        r"\bLT\b|\bLATA\b",
    "peso g/gr (15GR, 500G)":         r"\d+\s?[gG][rR]?\b",
    "peso kg":                        r"\d+\s?[kK][gG]\b",
    "S/ (sem, ex S/ACUCAR)":          r"\bS/\w+",
    "PET":                            r"\bPET\b",
    "garrafa/grf":                    r"\bGRF\b|\bGARRAFA\b",
}
for nome, rx in padroes.items():
    pct = q.str.contains(rx, regex=True, case=True).mean()
    pct_ci = q.str.contains(rx, regex=True, case=False).mean()
    w(f"   {nome:35s}: {pct_ci:6.1%} das queries")

# tokens mais frequentes
def tokens(serie):
    cnt = Counter()
    for t in serie:
        cnt.update(re.findall(r"[A-Za-zÀ-ÿ0-9/]+", t.lower()))
    return cnt

w()
w("top 30 tokens nas queries:")
for tok, n in tokens(q).most_common(30):
    w(f"   {n:6d}  {tok}")
w()
w("top 30 tokens no catálogo (product_name):")
for tok, n in tokens(c).most_common(30):
    w(f"   {n:6d}  {tok}")

# -------------------------------------------------------- 8. casos desafiadores
w()
w("=" * 70)
w("8. CASOS DESAFIADORES")
w("=" * 70)
w("--- 8a. Queries muito curtas (1-2 palavras) — pouco conteúdo discriminativo:")
curtas = queries[q.str.split().str.len() <= 2]
w(f"   total: {curtas.shape[0]} ({curtas.shape[0]/queries.shape[0]:.1%})")
for t in curtas["text"].head(10):
    w(f"   '{t}'")

w()
w("--- 8b. Ambiguidade no catálogo: mesmo produto-base em volumes/versões diferentes")
# remove volumes/pesos do nome e agrupa: grupos grandes = alta ambiguidade
def sem_volume(nome_prod):
    s = nome_prod.lower()
    s = re.sub(r"\d+[.,]?\d*\s?(ml|l|litros?|g|gr|kg|un|unidades?)\b", " ", s)
    s = re.sub(r"\d+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

catalog["_base"] = catalog["product_name"].map(sem_volume)
grupos = catalog.groupby("_base").size().sort_values(ascending=False)
w(f"   nomes-base com mais de 1 variação: {(grupos > 1).sum()} "
  f"({catalog[catalog['_base'].isin(grupos[grupos > 1].index)].shape[0]} produtos envolvidos)")
w("   maiores grupos de variações (mesmo nome-base):")
for base, n in grupos.head(8).items():
    exemplos = catalog[catalog["_base"] == base]["product_name"].head(3).tolist()
    w(f"   {n:3d}x  '{base}'  ex: {exemplos}")

w()
w("--- 8c. Possíveis NO_MATCH: queries cujos tokens quase não aparecem no catálogo")
vocab_cat = set(tokens(c).keys())
def cobertura(txt):
    toks = re.findall(r"[A-Za-zÀ-ÿ]+", txt.lower())
    toks = [t for t in toks if len(t) > 2]
    if not toks:
        return 1.0
    return sum(t in vocab_cat for t in toks) / len(toks)

queries["_cob"] = q.map(cobertura)
baixa = queries[queries["_cob"] <= 0.34]
w(f"   queries com ≤1/3 dos tokens no vocabulário do catálogo: {baixa.shape[0]} "
  f"({baixa.shape[0]/queries.shape[0]:.1%})")
for t in baixa["text"].head(12):
    w(f"   '{t}'")

w()
w("--- 8d. Exemplos aleatórios de queries (amostra para leitura humana):")
for t in queries["text"].sample(15, random_state=42):
    w(f"   '{t}'")

rel.close()
print("Exploração concluída. Saídas em:", OUT)
