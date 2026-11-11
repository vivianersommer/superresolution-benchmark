# Decisões de implementação

Decisões tomadas durante a execução do experimento, com a justificativa de cada uma. Os
identificadores (D1.1, D2.1 etc.) indicam a fase em que a decisão foi tomada.

## Ambiente

- **D0.1 — Tudo em Python.** A ART-ANOVA foi implementada em Python, sem depender do R
  (ver D5.5).
- **D0.2 — Inferência em CPU.** Os modelos rodam no backend padrão do OpenCV DNN, em CPU,
  por coerência com a medição de tempo da fase 2.

## Fase 1 — Dados e amostragem

- **D1.1 — Imagens cinza salvas em RGB são excluídas.** Imagens com R = G = B em todos os
  pixels são tratadas como não coloridas. Imagens quase cinza (11) continuam elegíveis.
- **D1.2 — Procedimento de sorteio.** As elegíveis são ordenadas por nome e sorteadas com
  `numpy.random.default_rng(42).choice(..., replace=False)`. A ordenação torna o sorteio
  independente da ordem do sistema de arquivos.
- **D1.3 — Decodificação.** Os JPEG são lidos com o OpenCV (`IMREAD_UNCHANGED`), o mesmo
  decodificador do resto do pipeline. As HR são salvas em PNG.
- **D1.4 — Imagens geradas fora do git.** As HR em PNG (49 MB) e os benchmarks (cerca de
  250 MB) não são versionados. Como as HR são regeneradas de forma determinística, os CSV
  com SHA-256 bastam para conferir a regeneração.
- **D1.5 — Fonte de Set5/Set14.** O pacote dos autores do EDSR, porque traz as LR geradas
  pelo MATLAB e permite testar a reimplementação da redução bicúbica.

## Fase 2 — Degradação e reconstrução

- **D2.1 — Bicúbica do experimento.** O método "bicúbica" é o `cv2.resize` com
  `INTER_CUBIC` (a = −0,75). O critério de validação de 0,1 dB no Set5 vale para o pipeline
  de métricas, testado com a bicúbica do MATLAB (`imresize`, a = −0,5). A bicúbica do
  OpenCV rende +0,31 dB (2×) e +0,21 dB (4×) acima da do MATLAB, o que torna a referência
  da H2 um pouco mais forte e o teste mais conservador.
- **D2.2 — Escalas.** Só 2× e 4×. A escala 3× exigiria outros modelos e não existe para
  o LapSRN.
- **D2.3 — Medição de tempo.** Só a chamada de reconstrução é medida (`perf_counter_ns`),
  depois de uma execução de aquecimento por modelo e escala, com todas as threads do OpenCV
  (16). Os tempos absolutos têm ruído; as ordens de grandeza entre métodos são robustas.

## Fase 3 — Métricas e validação

- **D3.1 — Canal Y sem arredondamento.** O Y é calculado em float64, como no BasicSR. A
  diferença para o Y arredondado é de 0,01 a 0,03 dB no Set5. Configurável em
  `metricas.y_round`.
- **D3.2 — Imagem de um canal.** A imagem `bridge` do Set14 vira BGR com três canais
  iguais antes da degradação e da reconstrução.
- **D3.3 — Critério de aceitação.** Usa a bicúbica do MATLAB (D2.1). A bicúbica do OpenCV
  aparece na tabela de validação só como informação.
- **D3.4 — Mod-crop.** A validação usa HR com mod-crop por s, como o pacote EDSR e os
  códigos de avaliação dos artigos. No COCO, o mod-crop é 12, para servir às duas escalas.
- **D3.5 — Diagnóstico dos pesos.** `f3_diagnostico.py` separa erro de pipeline de
  limitação dos pesos do OpenCV. Ele não altera nenhum dado da análise principal.

## Fase 4 — Estatística descritiva

- **D4.1 — Normalidade.** Avaliada nas diferenças "método − bicúbica" (os 6 contrastes
  com a referência), não nos 21 pares do post hoc.
- **D4.2 — Quantis e ICs.** Quantis com interpolação linear (tipo 7). IC da mediana por
  bootstrap percentil.

## Fase 5 — Inferência

- **D5.1 — Estrutura aleatória do modelo misto.** O modelo principal usa
  `(1|imagem) + (1|imagem:escala)`, porque os 7 métodos de cada imagem × escala partem da
  mesma LR (desenho do tipo split-plot). O modelo simplificado, só com `(1|imagem)`, é
  reportado como sensibilidade. AIC e resíduos confirmam a escolha.
- **D5.2 — Contrastes da H3.** Testados para os 6 métodos não-referência (4 modelos +
  vizinho + Lanczos), com Holm entre os 6 por métrica e procedimento. A H2 usa só os 4
  modelos.
- **D5.3 — Postos dentro de bloco.** Ajustados por OLS com erro-padrão robusto agrupado
  por imagem, com o teste de Wald conjunto, porque o modelo misto equivalente é degenerado.
- **D5.4 — Wilcoxon das diferenças das diferenças.** Teste não paramétrico direto do
  contraste de interação da H3: [(m − bic)₄ₓ − (m − bic)₂ₓ] por imagem.
- **D5.5 — ART em Python.** Alinhamento de Wobbrock et al. (2011), postos médios e o mesmo
  modelo misto. Os contrastes de interação usam as respostas alinhadas para a interação,
  como no ART-C.
- **D5.6 — p do Wilcoxon.** Aproximação normal com correção de empates (n = 100), com
  zeros descartados. Não houve diferenças nulas nos contrastes da H2.
- **D5.7 — Comparações de modelos.** Estruturas aleatórias comparadas por REML; interação
  testada por razão de verossimilhança com ajustes por ML; contrastes e EPs do ajuste REML
  (convenção do `lme4`/`emmeans`).
- **D5.8 — Otimizadores.** Vários otimizadores são tentados, e mantém-se o ajuste
  convergido de maior verossimilhança.
- **D5.9 — Correção múltipla nos contrastes.** Holm em todas as famílias (métrica × efeito
  × procedimento, 6 métodos).
