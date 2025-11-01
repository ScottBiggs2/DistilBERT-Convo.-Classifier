# Model Comparison Summary

## Average Inference Speed

| Model | Avg. Inference Time (ms) |
|---|---|| distilbert_onnx | 808.84 |
| distilbert_onnx_quant | 684.78 |
| distilbert_full | 304.22 |
| bert_onnx | 1541.32 |
| bert_onnx_quant | 1365.18 |
| bert_full | 584.57 |
| llama3.1_8b_groq | 0.00 |
| gemini_2.5_flash | 496.80 |

## Average Tokenization and Model Time (ms)

| Model | Tokenization (ms) | Model (ms) |
|---|---:|---:|
| distilbert_onnx | 3.10 | 805.46 |
| distilbert_onnx_quant | 2.80 | 681.80 |
| distilbert_full | 3.69 | 300.42 |
| bert_onnx | 2.62 | 1538.40 |
| bert_onnx_quant | 3.04 | 1361.94 |
| bert_full | 3.89 | 580.55 |
| llama3.1_8b_groq | N/A | N/A |
| gemini_2.5_flash | N/A | N/A |

![Avg tokenization time](avg_token_time.png)

![Avg model time](avg_model_time.png)

## Label Distribution per Model

### distilbert_onnx

| distilbert_onnx_pred   |   count |
|:-----------------------|--------:|
| A                      |     292 |
| B                      |     155 |
| C                      |      75 |
| D                      |     167 |
| E                      |      99 |
| F                      |      34 |
| G                      |      22 |
| H                      |      14 |
| I                      |      34 |
| J                      |      56 |
| K                      |      17 |
| L                      |      26 |
| M                      |       9 |

![Label Distribution for distilbert_onnx](distilbert_onnx_distribution.png)

### distilbert_onnx_quant

| distilbert_onnx_quant_pred   |   count |
|:-----------------------------|--------:|
| A                            |     292 |
| B                            |     155 |
| C                            |      75 |
| D                            |     167 |
| E                            |      99 |
| F                            |      34 |
| G                            |      22 |
| H                            |      14 |
| I                            |      34 |
| J                            |      56 |
| K                            |      17 |
| L                            |      26 |
| M                            |       9 |

![Label Distribution for distilbert_onnx_quant](distilbert_onnx_quant_distribution.png)

### distilbert_full

| distilbert_full_pred   |   count |
|:-----------------------|--------:|
| A                      |     292 |
| B                      |     155 |
| C                      |      75 |
| D                      |     167 |
| E                      |      99 |
| F                      |      34 |
| G                      |      22 |
| H                      |      14 |
| I                      |      34 |
| J                      |      56 |
| K                      |      17 |
| L                      |      26 |
| M                      |       9 |

![Label Distribution for distilbert_full](distilbert_full_distribution.png)

### bert_onnx

| bert_onnx_pred   |   count |
|:-----------------|--------:|
| A                |     301 |
| B                |     122 |
| C                |      79 |
| D                |     162 |
| E                |     116 |
| F                |      32 |
| G                |      26 |
| H                |      13 |
| I                |      32 |
| J                |      58 |
| K                |      21 |
| L                |      26 |
| M                |      12 |

![Label Distribution for bert_onnx](bert_onnx_distribution.png)

### bert_onnx_quant

| bert_onnx_quant_pred   |   count |
|:-----------------------|--------:|
| A                      |     301 |
| B                      |     122 |
| C                      |      79 |
| D                      |     162 |
| E                      |     116 |
| F                      |      32 |
| G                      |      26 |
| H                      |      13 |
| I                      |      32 |
| J                      |      58 |
| K                      |      21 |
| L                      |      26 |
| M                      |      12 |

![Label Distribution for bert_onnx_quant](bert_onnx_quant_distribution.png)

### bert_full

| bert_full_pred   |   count |
|:-----------------|--------:|
| A                |     301 |
| B                |     122 |
| C                |      79 |
| D                |     162 |
| E                |     116 |
| F                |      32 |
| G                |      26 |
| H                |      13 |
| I                |      32 |
| J                |      58 |
| K                |      21 |
| L                |      26 |
| M                |      12 |

![Label Distribution for bert_full](bert_full_distribution.png)

### llama3.1_8b_groq

| llama3.1_8b_groq_pred   |   count |
|:------------------------|--------:|
| ERROR                   |    1000 |

![Label Distribution for llama3.1_8b_groq](llama3.1_8b_groq_distribution.png)

### gemini_2.5_flash

| gemini_2.5_flash_pred   |   count |
|:------------------------|--------:|
| A                       |     309 |
| B                       |     123 |
| C                       |     149 |
| D                       |     157 |
| E                       |      88 |
| F                       |      27 |
| G                       |      27 |
| H                       |      12 |
| I                       |      29 |
| J                       |      39 |
| K                       |      18 |
| L                       |       7 |
| M                       |      15 |

![Label Distribution for gemini_2.5_flash](gemini_2.5_flash_distribution.png)

## Inter-Model Agreement Matrix

Percentage of times models agreed on the prediction.

|                       |   distilbert_onnx |   distilbert_onnx_quant |   distilbert_full |   bert_onnx |   bert_onnx_quant |   bert_full |   llama3.1_8b_groq |   gemini_2.5_flash |
|:----------------------|------------------:|------------------------:|------------------:|------------:|------------------:|------------:|-------------------:|-------------------:|
| distilbert_onnx       |           100.00% |                 100.00% |           100.00% |      83.60% |            83.60% |      83.60% |              0.00% |             75.00% |
| distilbert_onnx_quant |           100.00% |                 100.00% |           100.00% |      83.60% |            83.60% |      83.60% |              0.00% |             75.00% |
| distilbert_full       |           100.00% |                 100.00% |           100.00% |      83.60% |            83.60% |      83.60% |              0.00% |             75.00% |
| bert_onnx             |            83.60% |                  83.60% |            83.60% |     100.00% |           100.00% |     100.00% |              0.00% |             76.90% |
| bert_onnx_quant       |            83.60% |                  83.60% |            83.60% |     100.00% |           100.00% |     100.00% |              0.00% |             76.90% |
| bert_full             |            83.60% |                  83.60% |            83.60% |     100.00% |           100.00% |     100.00% |              0.00% |             76.90% |
| llama3.1_8b_groq      |             0.00% |                   0.00% |             0.00% |       0.00% |             0.00% |       0.00% |            100.00% |              0.00% |
| gemini_2.5_flash      |            75.00% |                  75.00% |            75.00% |      76.90% |            76.90% |      76.90% |              0.00% |            100.00% |

A detailed side-by-side comparison has been saved to [side_by_side_comparison.csv](side_by_side_comparison.csv).
