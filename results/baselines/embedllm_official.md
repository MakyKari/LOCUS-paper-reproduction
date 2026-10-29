# EmbedLLM under the official protocol

lr 0.0001, 50 epochs, batch 2048; test hit@1 in %, mean over seeds.

`test@50` = official (last epoch, no selection); `test@best_val` = same fit, epoch chosen by val hit@1.

| variant | shuffle | anchors | test@50 | test@best_val | best_val_epoch | paper |
|---|---|---|---|---|---|---|
| embedllm | False | 128 | 55.49 | 55.83 | 35.67 |  |
| embedllm | False | 256 | 60.37 | 60.29 | 49.67 | 58.80 |
| embedllm | False | 512 | 61.02 | 60.83 | 45.33 | 59.47 |
| embedllm | False | 768 | 61.44 | 61.43 | 48.00 |  |
| embedllm | False | 1024 | 62.77 | 62.70 | 48.00 | 59.60 |
| embedllm | False | 2048 | 64.47 | 64.23 | 42.00 |  |
| embedllm | True | 128 | 55.48 | 55.91 | 35.33 |  |
| embedllm | True | 256 | 59.92 | 59.92 | 48.00 | 58.80 |
| embedllm | True | 512 | 61.47 | 61.18 | 43.67 | 59.47 |
| embedllm | True | 768 | 61.72 | 61.64 | 43.67 |  |
| embedllm | True | 1024 | 62.94 | 62.94 | 47.67 | 59.60 |
| embedllm | True | 2048 | 64.23 | 64.67 | 44.67 |  |
| embedllm+1 | False | 128 | 58.34 | 58.26 | 49.67 |  |
| embedllm+1 | False | 256 | 60.46 | 60.53 | 40.67 | 59.17 |
| embedllm+1 | False | 512 | 62.19 | 62.79 | 42.67 | 61.57 |
| embedllm+1 | False | 768 | 61.83 | 62.09 | 37.33 |  |
| embedllm+1 | False | 1024 | 62.12 | 62.93 | 33.33 | 59.73 |
| embedllm+1 | False | 2048 | 64.37 | 64.60 | 21.33 |  |
| embedllm+1 | True | 128 | 58.44 | 58.34 | 47.67 |  |
| embedllm+1 | True | 256 | 61.10 | 61.46 | 46.33 | 59.17 |
| embedllm+1 | True | 512 | 60.54 | 62.49 | 31.33 | 61.57 |
| embedllm+1 | True | 768 | 60.80 | 61.86 | 27.00 |  |
| embedllm+1 | True | 1024 | 61.47 | 62.93 | 18.33 | 59.73 |
| embedllm+1 | True | 2048 | 64.04 | 64.68 | 18.00 |  |
| embedllm+2 | False | 128 | 57.88 | 58.24 | 42.67 |  |
| embedllm+2 | False | 256 | 60.74 | 60.81 | 33.67 | 59.17 |
| embedllm+2 | False | 512 | 61.52 | 61.86 | 45.33 | 61.57 |
| embedllm+2 | False | 768 | 61.07 | 61.29 | 34.67 |  |
| embedllm+2 | False | 1024 | 62.27 | 63.03 | 28.33 | 59.73 |
| embedllm+2 | False | 2048 | 64.14 | 64.80 | 24.33 |  |
| embedllm+2 | True | 128 | 58.77 | 58.64 | 42.33 |  |
| embedllm+2 | True | 256 | 59.61 | 60.77 | 29.33 | 59.17 |
| embedllm+2 | True | 512 | 60.11 | 62.18 | 28.33 | 61.57 |
| embedllm+2 | True | 768 | 60.00 | 62.46 | 22.33 |  |
| embedllm+2 | True | 1024 | 60.72 | 63.26 | 20.00 | 59.73 |
| embedllm+2 | True | 2048 | 61.18 | 65.24 | 11.67 |  |
