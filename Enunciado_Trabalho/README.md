# Enunciado T2: Matching de Produtos — Índice dos Documentos

Documentos extraídos do enunciado original (`enunciado_extensionista.pdf`) do Prof. Me. Otávio Parraga, organizados como contexto para a execução de cada etapa do trabalho.

## Documentos gerais

| Pasta | Documento | Conteúdo |
|---|---|---|
| `00_Contexto/` | [contexto.md](00_Contexto/contexto.md) | Problema, empresa parceira, exemplos de queries, recursos locais |
| `01_Objetivos/` | [objetivos.md](01_Objetivos/objetivos.md) | Objetivo geral, as 7 etapas, restrições metodológicas, prazo |
| `02_Especificacoes_Dados/` | [especificacoes_dados.md](02_Especificacoes_Dados/especificacoes_dados.md) | Os 4 CSVs, colunas, dimensões verificadas, protocolo de uso |
| `03_Metricas_e_Avaliacao/` | [metricas_e_avaliacao.md](03_Metricas_e_Avaliacao/metricas_e_avaliacao.md) | P@1, MRR@5, R@5 (fórmulas e exemplos), protocolo validação × teste |
| `04_Entregaveis/` | [entregaveis.md](04_Entregaveis/entregaveis.md) | README.txt, código, relatório (6 seções), checklist, prazo 30/06 |
| `99_Enunciado_Completo/` | [enunciado_completo.md](99_Enunciado_Completo/enunciado_completo.md) | Transcrição integral do PDF |

## As 7 etapas do trabalho

| Pasta | Documento | Etapa |
|---|---|---|
| `Etapa01_Exploração dos dados/` | [etapa_1_exploracao_dos_dados.md](<Etapa01_Exploração dos dados/etapa_1_exploracao_dos_dados.md>) | Exploração de `catalog.csv` e `queries.csv` |
| `Etapa02_ Abordagem 1 — NLP Clássico/` | [etapa_2_nlp_classico.md](<Etapa02_ Abordagem 1 — NLP Clássico/etapa_2_nlp_classico.md>) | Pipeline TF-IDF/BM25 + métricas na validação |
| `Etapa03_ Abordagem 2 — NLP com Deep Learning/` | [etapa_3_deep_learning.md](<Etapa03_ Abordagem 2 — NLP com Deep Learning/etapa_3_deep_learning.md>) | Pipeline com rede neural (embeddings ou LLM) + métricas na validação |
| `Etapa04_Comparação das abordagens/` | [etapa_4_comparacao_das_abordagens.md](<Etapa04_Comparação das abordagens/etapa_4_comparacao_das_abordagens.md>) | Tabela comparativa e discussão de tradeoffs |
| `Etapa05_Avaliação final/` | [etapa_5_avaliacao_final.md](<Etapa05_Avaliação final/etapa_5_avaliacao_final.md>) | Métricas definitivas sobre `queries_test.csv` |
| `Etapa06_Análise qualitativa/` | [etapa_6_analise_qualitativa.md](<Etapa06_Análise qualitativa/etapa_6_analise_qualitativa.md>) | Acertos, erros, ambíguos e NO_MATCH |
| `Etapa07_Relatório e apresentação/` | [etapa_7_relatorio_e_apresentacao.md](<Etapa07_Relatório e apresentação/etapa_7_relatorio_e_apresentacao.md>) | Relatório, README, código e apresentação oral |

## Recursos do projeto

- **Dados:** `C:\GitHub_Rafa\Case_Nubo_Aprendizado_Maquina\Dados_a_trabalhar` (`catalog.csv`, `queries.csv`, `queries_val.csv`, `queries_test.csv`)
- **Slides de Redes Neurais:** `C:\GitHub_Rafa\Case_Nubo_Aprendizado_Maquina\Slides_professor\Redes Neurais`
- **Enunciado original:** `enunciado_extensionista.pdf` (nesta pasta)
