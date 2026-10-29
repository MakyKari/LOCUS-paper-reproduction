# Stage 12: timing

GPU: NVIDIA GeForce RTX 3050 Laptop GPU, torch 2.6.0+cu118, dtype torch.float32; median of 50 runs after 10 warm-ups, ms. Paper: V100.

## Encoder (ms)

| models | n=256 | n=512 | n=1024 | n=2048 | n=4096 |
|---|---|---|---|---|---|
| 1 | 4.44 | 3.61 | 3.56 | 3.60 | 5.09 |
| 16 | 5.12 | 8.02 | 15.31 | 31.48 | 62.02 |
| 32 | 8.43 | 15.30 | 29.78 | 60.88 | 123.26 |
| 64 | 16.00 | 29.76 | 59.40 | 123.33 | 243.92 |
| 96 | 23.48 | 44.65 | 89.79 | 185.21 | 363.51 |
| 112 | 27.52 | 52.65 | 105.07 | 218.85 | 422.67 |

- 112 x 4096: 422.7 ms (paper ≈104 ms), model chunk 28.
- |M| = 112: time ratio n=4096 / n=256 = 15.4 (linear would be 16).
- |M| = 1: time ratio n=4096 / n=256 = 1.1.
- Chunked over models (did not fit at once): 96x2048 in chunks of 48, 112x2048 in chunks of 56, 64x4096 in chunks of 32, 96x4096 in chunks of 24, 112x4096 in chunks of 28.

## Decoder, 112 models x 4096 queries

- GPU only: 18.5 ms (query chunk 4096); paper ≈18 ms.
- Through the authors' score_queries_against_models (chunks 1024 x 256, copies to host): 21.2 ms.
