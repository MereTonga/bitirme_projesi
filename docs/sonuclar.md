| Model | Ayar | ROC-AUC | PR-AUC | F1 | Recall | Precision | Bal. Acc. | Accuracy | CV AUC ± std |
|---|---|---|---|---|---|---|---|---|---|
| Gradient Boosting | Optuna (20 trial) | **0.682** | 0.617 | 0.538 | 0.535 | 0.541 | 0.623 | 0.642 | 0.678 ± 0.021 |
| Random Forest | Optuna (20 trial) | **0.681** | 0.605 | 0.523 | 0.484 | 0.569 | 0.625 | 0.655 | 0.694 ± 0.017 |
| Random Forest | varsayılan | **0.675** | 0.592 | 0.524 | 0.469 | 0.594 | 0.632 | 0.668 | — |
| XGBoost | Optuna (20 trial) | **0.674** | 0.603 | 0.558 | 0.617 | 0.510 | 0.619 | 0.619 | 0.684 ± 0.012 |
| ANN (PyTorch) | varsayılan (64-32) | **0.673** | 0.603 | 0.544 | 0.562 | 0.527 | 0.620 | 0.633 | — |
| Gradient Boosting | varsayılan | **0.668** | 0.600 | 0.535 | 0.539 | 0.531 | 0.617 | 0.634 | — |
| XGBoost | varsayılan | **0.666** | 0.608 | 0.502 | 0.488 | 0.516 | 0.598 | 0.622 | — |
| LightGBM | Optuna (20 trial) | **0.664** | 0.594 | 0.515 | 0.477 | 0.560 | 0.618 | 0.649 | 0.684 ± 0.018 |
| LightGBM | varsayılan | **0.659** | 0.590 | 0.528 | 0.508 | 0.551 | 0.621 | 0.646 | — |
| SVM | varsayılan | **0.654** | 0.568 | 0.523 | 0.535 | 0.511 | 0.604 | 0.619 | — |
| SVM | Optuna (20 trial) | **0.652** | 0.571 | 0.523 | 0.551 | 0.498 | 0.598 | 0.608 | 0.701 ± 0.018 |
| AdaBoost | Optuna (20 trial) | **0.628** | 0.542 | 0.492 | 0.484 | 0.500 | 0.587 | 0.610 | 0.667 ± 0.020 |
| KNN | Optuna (20 trial) | **0.626** | 0.540 | 0.519 | 0.559 | 0.485 | 0.589 | 0.596 | 0.659 ± 0.015 |
| KNN | varsayılan | **0.617** | 0.494 | 0.477 | 0.492 | 0.463 | 0.564 | 0.579 | — |
| Naive Bayes | varsayılan | **0.609** | 0.484 | 0.454 | 0.445 | 0.463 | 0.558 | 0.582 | — |
| Naive Bayes | Optuna (20 trial) | **0.609** | 0.484 | 0.454 | 0.445 | 0.463 | 0.558 | 0.582 | 0.578 ± 0.024 |
| Decision Tree | Optuna (20 trial) | **0.601** | 0.518 | 0.402 | 0.324 | 0.529 | 0.570 | 0.624 | 0.627 ± 0.022 |
| Decision Tree | varsayılan | **0.585** | 0.441 | 0.503 | 0.519 | 0.487 | 0.585 | 0.599 | — |
| AdaBoost | varsayılan | **0.566** | 0.457 | 0.495 | 0.582 | 0.431 | 0.545 | 0.537 | — |
| Logistic Regression | Optuna (20 trial) | **0.553** | 0.470 | 0.479 | 0.555 | 0.421 | 0.534 | 0.529 | 0.473 ± 0.016 |
| Logistic Regression | varsayılan | **0.547** | 0.460 | 0.474 | 0.547 | 0.418 | 0.530 | 0.526 | — |

*Üretim: `python scripts/sonuc_tablosu.py` — 2026-11-04. Test seti: %20 stratified (n=656), random_state=42; median imputer, RobustScaler, SMOTE yalnızca eğitim setinde. Test metrikleri tek seferlik; CV AUC Optuna aramasının eğitim setindeki 5-fold ortalamasıdır (test seti aramaya girmez). Sıralama test ROC-AUC'ye göre.*

### Bölme değişkenliği (varsayılan parametreler, 10 farklı stratified bölme)

| Model | Test ROC-AUC ort ± std | min – max |
|---|---|---|
| SVM | **0.683** ± 0.016 | 0.659 – 0.706 |
| Random Forest | **0.680** ± 0.020 | 0.655 – 0.711 |
| XGBoost | **0.669** ± 0.022 | 0.641 – 0.718 |
| LightGBM | **0.658** ± 0.019 | 0.624 – 0.689 |
| Gradient Boosting | **0.645** ± 0.020 | 0.615 – 0.677 |
| KNN | **0.617** ± 0.021 | 0.594 – 0.666 |
| Decision Tree | **0.580** ± 0.017 | 0.548 – 0.605 |
| Naive Bayes | **0.577** ± 0.027 | 0.539 – 0.639 |
| AdaBoost | **0.557** ± 0.021 | 0.529 – 0.596 |
| Logistic Regression | **0.497** ± 0.013 | 0.478 – 0.521 |

*Her bölmede aynı preprocessing (median, RobustScaler, SMOTE yalnızca eğitimde), random_state 0…9. ANN bu tabloda yok (GPU/epoch süresi).*
