# Comparação estatística pareada entre super-resolução por aprendizado profundo e interpolação clássica

Código, dados de apoio e resultados do artigo *"Além da média do PSNR: comparação
estatística pareada entre super-resolução por aprendizado profundo e interpolação
clássica"* (Viviane da Rosa Sommer, UTFPR, 2026).

**Pergunta.** Implementações pré-treinadas de quatro modelos de aprendizado profundo
(EDSR, ESPCN, FSRCNN e LapSRN, distribuídas com o OpenCV) reconstroem imagens melhor que a
interpolação clássica (bicúbica, vizinho mais próximo e Lanczos)? E essa vantagem se
mantém quando o fator de escala passa de 2× para 4×?

**Desenho.**
- 100 imagens do COCO, sorteadas com semente 42, × 7 métodos × 2 escalas = 1.400
  reconstruções.
- A LR é gerada pelo `imresize` bicúbico do MATLAB, com antialias.
- Métricas: PSNR-Y, SSIM-Y e LPIPS.
- Inferência: Friedman, Wilcoxon pareado com correção de Holm, modelo linear misto com
  interceptos de imagem e de imagem × escala, e ART como robustez.

## Principais resultados

| Hipótese | Resultado |
| --- | --- |
| H1 — os 7 métodos diferem | Sim, em todas as métricas e escalas (Friedman, p < 10⁻⁹², W de Kendall de 0,74 a 0,91). Uma ANOVA não pareada nos mesmos dados não detecta diferença em PSNR nem em SSIM. |
| H2 — modelos × bicúbica | Em 2×, os 4 modelos superam a bicúbica nas 3 métricas. Em 4×, só o EDSR mantém vantagem em PSNR e SSIM; todos seguem melhores em LPIPS. |
| H3 — interação método × escala | Significativa nas 3 métricas. A vantagem em PSNR/SSIM diminui de 2× para 4× (EDSR: +1,02 → +0,43 dB) e, em LPIPS, aumenta. |

As tabelas completas estão em [`results/`](results/) e as figuras em
[`results/figs/`](results/figs/).

## Estrutura

```
├── config.yaml            todos os parâmetros (semente, escalas, métodos, caminhos, figuras)
├── requirements.txt       versões fixas dos pacotes
├── run_all.py             refaz as fases 1 a 5 a partir das imagens
├── DECISOES.md            decisões de implementação e justificativas
├── src/
│   ├── common.py             leitura do config e resolução de caminhos
│   ├── plot_style.py         paleta, fonte e exportação das figuras (PNG/PDF, 300 dpi)
│   ├── imresize_matlab.py    imresize bicúbico do MATLAB (idêntico bit a bit às LR de referência)
│   ├── f0_ambiente.py        fase 0: ambiente e manifesto dos modelos
│   ├── f1_amostragem.py      fase 1: inventário, elegibilidade, sorteio, mod-crop, benchmarks
│   ├── f2_reconstrucao.py    fase 2: LR, 7 métodos, tempos, grade visual
│   ├── f3_metricas.py        fase 3: PSNR-Y, SSIM-Y, LPIPS, MSE-Y; validação no Set5/Set14
│   ├── f3_diagnostico.py     fase 3: diagnóstico do desempenho dos pesos do OpenCV
│   ├── f4_descritiva.py      fase 4: descritiva, ICs bootstrap, normalidade, figuras
│   └── f5_inferencia.py      fase 5: Friedman, post hoc, H2, modelo misto, ART, poder
├── tests/                 testes por fase (pytest)
├── models/                pesos pré-treinados (.pb) do OpenCV + MANIFEST.csv (SHA-256)
├── dataset/
│   └── dataset_COCO_2000/    2.000 imagens JPEG do COCO (base de sorteio)
├── data/
│   ├── DATA_CARD.md          origem, critérios de elegibilidade, exclusões, licenças
│   ├── inventario_coco.csv   as 2.000 imagens, com o motivo de exclusão
│   ├── amostra.csv           as 100 sorteadas, com SHA-256 do JPEG e do PNG cortado
│   └── benchmarks_manifest.csv  SHA-256 dos arquivos de Set5/Set14
└── results/
    ├── metrics_long.csv      1.400 linhas: image, method, family, scale, psnr_y, ssim_y, lpips, mse_y
    ├── runtime_long.csv      tempo de reconstrução por imagem × método × escala
    ├── validacao_*.csv       validação no Set5/Set14 e diagnóstico
    ├── descritiva.csv, medianas_ic.csv, normalidade_diferencas.csv
    ├── inferencia_*.csv/.md  H1, post hoc, H2, H3 e comparação das estruturas aleatórias
    ├── modelo_misto.txt, art_anova.txt, poder.txt
    ├── ambiente.json         SO, CPU, versões dos pacotes
    └── figs/                 figuras em PNG e PDF
```

As pastas `data/hr/`, `data/benchmarks/` e `outputs/` são geradas pelos scripts e ficam
fora do git.

## Instalação

Requer Python 3.14 (testado em Ubuntu 26.04, CPU Intel i7-13620H). Não usa GPU.

```bash
python3.14 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt   # torch CPU via --extra-index-url
```

Na primeira execução há dois downloads automáticos:
- o pacote de benchmarks dos autores do EDSR, com Set5 e Set14 (cerca de 250 MB, SHA-256
  conferido);
- os pesos da AlexNet usados pelo LPIPS (cerca de 233 MB, em `~/.cache/torch`).

## Como rodar

A partir da raiz do repositório:

```bash
.venv/bin/python run_all.py            # fases 1 a 5 (~25 min em CPU)
.venv/bin/python run_all.py --tests    # + testes no fim
.venv/bin/python run_all.py --from 3   # retoma a partir da fase 3
```

O tempo é dominado pelo EDSR 2× (cerca de 10 s por imagem em CPU). O log fica em
`outputs/run_all.log`. Qualquer erro interrompe a execução.

Cada fase também pode ser rodada sozinha, por exemplo `.venv/bin/python src/f3_metricas.py`.
A fase 0 registra o ambiente e não faz parte do `run_all.py`. Testes:
`.venv/bin/python -m pytest -q tests/`.

## Reprodutibilidade

- Semente 42 em todo passo aleatório (sorteio, bootstrap e jitter das figuras).
- SHA-256 conferidos para os modelos, os JPEG de origem, as HR cortadas e o pacote de
  benchmarks.
- A reconstrução é determinística, e a redução é idêntica à do MATLAB (testado).
- O `run_all.py` foi executado numa cópia limpa do repositório, com ambiente novo, e
  reproduziu os mesmos números. Só os tempos de execução variam, como esperado.

## Licença dos dados

As imagens do COCO têm licenças Creative Commons individuais, definidas pelos autores no
Flickr (ver [`data/DATA_CARD.md`](data/DATA_CARD.md)). Set5 e Set14 são baixados da fonte
original e não são redistribuídos aqui.
