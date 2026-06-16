# -*- coding: utf-8 -*-
"""
Pré-processamento textual compartilhado pelas Abordagens 1 e 2.
Regras derivadas da exploração da Etapa 1:
  - minúsculas, sem acento, sem pontuação;
  - termos de EMBALAGEM DO PEDIDO (C/6, 12X15, UND, CX, FD, PCT) são removidos
    porque descrevem o pedido, não o produto;
  - volumes/pesos são MANTIDOS e canonizados (1 litro -> 1l, 15 GR -> 15g),
    pois distinguem variações do mesmo produto (água 500ml × 2 litros);
  - stopwords leves; "com"/"sem" são MANTIDAS (distinguem com gás × sem gás).
"""
import re
import unicodedata

STOPWORDS = {
    "de", "do", "da", "dos", "das", "e", "em", "no", "na", "nos", "nas",
    "o", "a", "os", "as", "ao", "aos", "um", "uma", "para", "por",
}

# abreviações de embalagem do pedido (removidas por completo)
_PADROES_EMBALAGEM = [
    r"\bc/\s*\d+\b",                # c/6, c/ 12  (caixa com N)
    r"\bund\b", r"\bunid\b", r"\bun\b", r"\bunds\b",
    r"\bcx\b", r"\bfd\b", r"\bfardo\b", r"\bpct\b", r"\bemb\b",
]


def _remover_acentos(t: str) -> str:
    return unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()


def normalizar(texto: str) -> str:
    """Normaliza um texto (query ou nome de produto) para o matching."""
    t = texto.lower()
    t = _remover_acentos(t)
    t = t.replace(",", ".")                          # decimais: 1,5 -> 1.5
    t = re.sub(r"\bs/\s*", "sem ", t)                # s/acucar -> sem acucar
    # embalagem do pedido: 12x15 -> mantém só o conteúdo unitário (15)
    t = re.sub(r"\b\d+\s*x\s*(\d+(?:\.\d+)?)\b", r"\1", t)
    for rx in _PADROES_EMBALAGEM:
        t = re.sub(rx, " ", t)
    # unidades canônicas (sempre coladas ao número: 1 litro -> 1l)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*(litros?|lts?|l)\b", r"\1l", t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*ml\b", r"\1ml", t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*(quilos?|kgs?|kg)\b", r"\1kg", t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*(gramas?|grs?|gr|g)\b", r"\1g", t)
    t = re.sub(r"\blt\b", "lata", t)                 # LT sem número = lata
    t = re.sub(r"[^a-z0-9. ]", " ", t)               # pontuação -> espaço
    t = re.sub(r"\.(?=\D)|(?<=\D)\.", " ", t)        # ponto fora de número
    tokens = [tok for tok in t.split() if tok not in STOPWORDS]
    return " ".join(tokens)


def montar_documento(product_name: str, brand_name: str, usar_marca: bool) -> str:
    """Texto do catálogo a indexar; opcionalmente anexa a marca se ausente do nome."""
    doc = normalizar(product_name)
    if usar_marca and isinstance(brand_name, str):
        marca = normalizar(brand_name)
        if marca and marca not in doc:
            doc = f"{doc} {marca}"
    return doc
