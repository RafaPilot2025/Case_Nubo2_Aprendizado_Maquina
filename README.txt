T2: MATCHING DE PRODUTOS — Aprendizado de Maquina (Prof. Otavio Parraga)
=========================================================================

INTEGRANTES
-----------
- Rafael Magalhaes      — matricula: 25180166
- Pedro Martini Lehn    — matricula: 22280113

ESTRUTURA DO PROJETO
--------------------
Dados_a_trabalhar/            catalog.csv, queries.csv, queries_val.csv, queries_test.csv
Desenvolvimento/
  comum/                      preprocessamento.py (normalizacao) e avaliacao.py (P@1, MRR@5, R@5)
  Etapa01_Exploracao_dos_Dados/      exploracao_dados.py + resultados/
  Etapa02_NLP_Classico/              pipeline_classico.py (TF-IDF e BM25, 8 variacoes) + resultados/
  Etapa03_Deep_Learning/             embeddings_semanticos.py, rerank_gemini.py, rerank_claude.py + resultados/
  Etapa04_Comparacao/                comparacao_abordagens.py + resultados/
  Etapa05_Avaliacao_Final/           avaliacao_final_locais.py + resultados/ (metricas no teste)
  Etapa06_Analise_Qualitativa/       analise_qualitativa.py + resultados/ (inclui experimento NO_MATCH)
  Etapa07_Relatorio_e_Apresentacao/  RELATORIO.md, RELATORIO.pdf

INSTALACAO (Windows, Python 3.12+)
----------------------------------
1) Instalar dependencias:
     python -m pip install -r requirements.txt

2) Configurar as API keys como variaveis de ambiente (necessarias apenas
   para os scripts de LLM — rerank_claude.py, rerank_gemini.py e o
   experimento NO_MATCH da Etapa 6):

     setx ANTHROPIC_API_KEY "sk-ant-...sua-chave"
     setx GEMINI_API_KEY    "...sua-chave"        (opcional; so p/ reproduzir o experimento Gemini)

   As chaves sao lidas do ambiente; nenhuma chave fica no codigo.
   Abra um NOVO terminal apos o setx. Custo estimado para reproduzir a
   parte Claude (validacao + teste + NO_MATCH): ~US$ 0,85 em creditos.

REPRODUCAO DOS RESULTADOS (ordem)
---------------------------------
  cd Desenvolvimento
  python Etapa01_Exploracao_dos_Dados\exploracao_dados.py
  python Etapa02_NLP_Classico\pipeline_classico.py
  python Etapa03_Deep_Learning\embeddings_semanticos.py        (baixa ~1 GB de modelos na 1a vez)
  python Etapa03_Deep_Learning\rerank_claude.py zero val       (requer ANTHROPIC_API_KEY)
  python Etapa03_Deep_Learning\rerank_claude.py few  val
  python Etapa04_Comparacao\comparacao_abordagens.py
  python Etapa05_Avaliacao_Final\avaliacao_final_locais.py
  python Etapa03_Deep_Learning\rerank_claude.py few  test      (avaliacao final da Abordagem 2)
  python Etapa06_Analise_Qualitativa\analise_qualitativa.py    (requer ANTHROPIC_API_KEY)

  Opcional (experimento historico com a API gratuita do Gemini):
  python Etapa03_Deep_Learning\rerank_gemini.py zero|few       (requer GEMINI_API_KEY; sujeito
                                                                a cota de 20 req/dia por modelo)

OBSERVACOES
-----------
- Os scripts de LLM gravam checkpoints (*.jsonl) e podem ser retomados se interrompidos.
- queries_test.csv foi usado UMA unica vez por abordagem, apos congelar as
  configuracoes na validacao (detalhes na secao 3 do relatorio).
- Resultados finais no teste: Abordagem 1 (TF-IDF char) P@1=99,2%;
  Abordagem 2 (Claude few-shot) P@1=99,6%.
