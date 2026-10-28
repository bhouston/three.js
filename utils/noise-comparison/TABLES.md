# Reproduced comparison

Uniform field: standard deviation after arithmetic running average and wrapped sigma=1 Gaussian. Lower is better. Full precision is in results.json.

| Frames | White | FAST Spatial (Product) | FAST Spatio-Temporal (Separate) | Golden Ratio | 3LHW | Animated IGN |
|---|---|---|---|---|---|---|
| 1 | 0.0791102 | 0.0191279 | 0.0221912 | 0.0191279 | 0.0496676 | 0.0293557 |
| 2 | 0.0566971 | 0.01304 | 0.0129899 | 0.0274788 | 0.0283296 | 0.0197332 |
| 4 | 0.0398381 | 0.00845266 | 0.00681967 | 0.0198493 | 0.0186524 | 0.00991156 |
| 8 | 0.028783 | 0.00538188 | 0.0034544 | 0.0106018 | 0.0106894 | 0.00603109 |
| 16 | 0.0205297 | 0.00347396 | 0.00184849 | 0.00635346 | 0.00458983 | 0.00336372 |
| 32 | 0.0144189 | 0.00216621 | 0.000635486 | 0.00333969 | 0.00288429 | 0.00221593 |
| 64 | 0.0102317 | 0.00152805 | 0.000450213 | 0.00180239 | 0.00183678 | 0.00136672 |
| 128 | 0.007249 | 0.00107426 | 0.000314946 | 0.00100038 | 0.000998968 | 0.000939296 |
| 256 | 0.00512031 | 0.000759907 | 0.000222616 | 0.000512805 | 0.000478195 | 0.000602652 |
| 512 | 0.00365363 | 0.000546447 | 0.0001589 | 0.000275256 | 0.000292861 | 0.000277884 |
| 1024 | 0.00257676 | 0.000382102 | 0.000115059 | 0.000137979 | 0.00010856 | 0.000205356 |

## Nonlinear check

RMSE against constant probability after thresholding each sample, averaging 256 frames, and applying the same Gaussian. Unlike standard deviation, RMSE includes bias.

| Method | p=0.1 | p=0.5 | p=0.9 |
|---|---|---|---|
| White | 0.00533275 | 0.00897814 | 0.00535868 |
| FAST Spatial (Product) | 0.00279414 | 0.00270784 | 0.00279901 |
| FAST Spatio-Temporal (Separate) | 0.00303775 | 0.00269349 | 0.00299872 |
| Golden Ratio | 0.000749063 | 0.000816233 | 0.000744375 |
| 3LHW | 0.000731546 | 0.000407901 | 0.000740461 |
| Animated IGN | 0.00096324 | 0.000707378 | 0.000994659 |

## First 32 frames: spectrum diagnostics

Ratios of mean power: spatial frequencies 0<r<0.1 / r>0.3; temporal bins 1–3 / all non-DC bins. Lower indicates more suppression in those bands, not universally better rendering.

| Method | Spatial low/high | Temporal low/all |
|---|---|---|
| White | 0.988624 | 0.993352 |
| FAST Spatial (Product) | 0.00101288 | 0.961534 |
| FAST Spatio-Temporal (Separate) | 0.00212279 | 0.0958767 |
| Golden Ratio | 0.213005 | 0.26794 |
| 3LHW | 0.112514 | 0.267082 |
| Animated IGN | 0.0414616 | 0.235388 |
