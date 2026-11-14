| Scenario | AUC (mean ± sd) | ΔAUC vs control | Recall | Default rate |
|---|---|---|---|---|
| controle (nenhuma causa) | 0.779 ± 0.000 | +0.000 | 0.357 | 22.1% |
| só inflação (faturas) | 0.777 ± 0.000 | -0.002 | 0.373 | 22.1% |
| só mais atrasos (pay_1) | 0.772 ± 0.002 | -0.007 | 0.381 | 22.1% |
| só perfil mais jovem (age) | 0.778 ± 0.000 | -0.001 | 0.354 | 22.1% |
| só data drift (as 3 causas) | 0.768 ± 0.002 | -0.011 | 0.389 | 22.1% |
| só concept drift | 0.651 ± 0.004 | -0.128 | 0.232 | 34.2% |
| tudo (período 4) | 0.653 ± 0.004 | -0.125 | 0.283 | 30.6% |

Attribution of the AUC drop (30 repetitions): data drift -0.011, concept drift -0.128, interaction +0.014, total -0.125.