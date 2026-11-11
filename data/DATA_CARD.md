# Data card

Gerado na fase 1 (2026-10-29) por `src/f1_amostragem.py`; parâmetros em `config.yaml`
(blocos `amostragem` e `benchmarks`).

## 1. Base principal: subconjunto COCO (`dataset/dataset_COCO_2000/`)

**Origem.** São 2.000 JPEGs do Microsoft COCO (Lin et al., 2014). Não há registro de qual
split (train/val) nem de qual versão do COCO vieram, nem de como as 2.000 foram escolhidas.
Os nomes seguem o padrão de ID do COCO 2014/2017 (`000000xxxxxx.jpg`). Por isso, a
população de inferência do estudo é este subconjunto de 2.000 imagens.

**Licença.** As imagens do COCO vêm do Flickr, e cada uma tem a sua licença Creative
Commons, informada no campo `license` das anotações oficiais. As anotações do COCO são
CC BY 4.0, e o consórcio COCO não detém os direitos das imagens. A licença de cada imagem da
amostra **não foi verificada**, porque as anotações não estão no repositório. Uso no
experimento: pesquisa acadêmica, sem redistribuição das imagens.

**Formato.** Todas são JPEG, ou seja, a própria referência HR tem artefatos de compressão
(limitação discutida no artigo). Nenhuma tem tag EXIF de orientação
diferente de 1 (verificado). A leitura usa `cv2.IMREAD_UNCHANGED`, sem conversão de cor
nem rotação.

### Inventário (`data/inventario_coco.csv`)

| Item | Valor |
| --- | --- |
| Arquivos | 2.000 (todos JPEG legíveis) |
| 3 canais | 1.996 (dos quais 44 são cinza salva em RGB, R=G=B em todos os pixels) |
| 1 canal (cinza) | 4 |
| Menor lado (mín. / mediana / máx.) | 111 / 428 / 640 px |

### Critérios de elegibilidade e exclusões

1. Imagem colorida: 3 canais e **não** cinza salva em RGB (decisão D1.1, ver `DECISOES.md`).
2. Menor lado ≥ 256 px, para que a versão reduzida 4× tenha pelo menos 64 px.

Os critérios são aplicados nessa ordem, e cada imagem excluída recebe um único motivo:

| Motivo | Imagens |
| --- | --- |
| 1 canal | 4 |
| Cinza salva em RGB (R=G=B) | 44 |
| Menor lado < 256 px | 11 |
| **Elegíveis** | **1.941** |

Outras 11 imagens são "quase cinza" (diferença média de croma < 1 nível, mas não zero).
Elas continuam elegíveis, porque o critério exclui só a igualdade exata entre os canais.

## 2. Amostra (`data/amostra.csv`)

- **Desenho:** amostragem aleatória simples, sem reposição, de n = 100 entre as 1.941
  elegíveis.
- **Procedimento:** a lista de elegíveis é ordenada por nome de arquivo e sorteada com
  `numpy.random.default_rng(42).choice(1941, 100, replace=False)`. A coluna
  `ordem_sorteio` guarda a ordem do sorteio.
- **Colunas:** `ordem_sorteio, arquivo, largura, altura, sha256` (JPEG original) e
  `hr_arquivo, hr_largura, hr_altura, hr_sha256` (PNG cortado).

### HR com mod-crop (`data/hr/`, fora do git)

- Cada imagem é cortada pela direita e por baixo até dimensões divisíveis por 12, e a mesma
  HR serve para 2×, 3× e 4×. O corte máximo foi de 11 px em cada eixo.
- As HR são salvas em PNG (sem perda) com `cv2.imwrite`, na ordem de canais BGR do OpenCV.
- Dimensões da HR: largura de 324 a 636 px, altura de 300 a 636 px. O menor lado das HR é
  300 px, então a LR 4× tem pelo menos 75 px.
- Para regenerar, rode `src/f1_amostragem.py`. O SHA-256 de cada PNG fica em `amostra.csv`.

## 3. Benchmarks de validação (`data/benchmarks/`, fora do git)

| Base | Referência | Conteúdo usado | Observação |
| --- | --- | --- | --- |
| Set5 | Bevilacqua et al., BMVC 2012 | 5 HR + LR_bicubic X2/X3/X4 | todas RGB |
| Set14 | Zeyde et al., Curves and Surfaces 2010 | 14 HR + LR_bicubic X2/X3/X4 | `bridge.png` tem 1 canal (cinza) |

- **Fonte:** pacote `benchmark.tar` dos autores do EDSR (Lim et al., 2017),
  <https://cv.snu.ac.kr/research/EDSR/benchmark.tar> (Last-Modified: 2018-03-20), baixado
  em 2026-10-29.
- **SHA-256 do tar:** `80c21c333bbf6ceb5308b7243761f8284478274413a97b96f1d63e9045fd93e8`
  (conferido a cada execução).
- **Arquivos extraídos:** o SHA-256 de cada um está em `data/benchmarks_manifest.csv`.
- **Uso das LR:** as `LR_bicubic` do pacote foram geradas pelo `imresize` do MATLAB.
  Servem de referência para testar a reimplementação da redução bicúbica na fase 2.
- **Licença:** Set5 e Set14 não têm licença formal. São distribuídos pelos autores e
  amplamente redistribuídos para pesquisa acadêmica em super-resolução. O uso aqui é só
  de validação, sem redistribuição.
