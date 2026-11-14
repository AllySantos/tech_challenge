| Test | AUC (5-fold CV) |
|---|---|
| Predict `sex` from the model features | 0.649 ± 0.007 |
| Control: shuffled labels (should be ~0.5) | 0.498 ± 0.006 |

Gap vs shuffled-label control: **+0.152** AUC. Any reading of this gap (weak, moderate, notable) is a judgment call and is made in the README, not here.

Top features for predicting the protected attribute:

| Feature | Importance |
|---|---|
| limit_bal | 0.098 |
| pay_3 | 0.096 |
| age | 0.094 |
| education | 0.075 |
| marriage | 0.075 |