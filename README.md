# Handling Class Imbalance on Liver Cancer Dataset Using Random Forest

## Overview
This project addresses the **class imbalance problem** in a liver cancer dataset using Random Forest classifier. Three resampling techniques were compared to find the best approach for improving minority class detection.

## Methods
| Technique | Description |
|---|---|
| **Baseline** | Random Forest without resampling |
| **ROS** | Random Over-Sampling |
| **SMOTENC** | SMOTE for Numerical + Categorical features |
| **SMOTENC+ENN** | SMOTENC combined with Edited Nearest Neighbours |

## Results (95% Confidence Interval)
| Model | Precision | Recall | F1 | Balanced Accuracy |
|---|---|---|---|---|
| Baseline | 0.963 ± 0.004 | 0.590 ± 0.005 | 0.732 ± 0.004 | 0.792 ± 0.003 |
| ROS | 0.893 ± 0.004 | 0.745 ± 0.006 | 0.812 ± 0.004 | 0.860 ± 0.003 |
| SMOTENC | 0.715 ± 0.003 | 0.781 ± 0.004 | 0.746 ± 0.003 | 0.847 ± 0.002 |
| SMOTENC+ENN | 0.754 ± 0.003 | 0.679 ± 0.004 | 0.715 ± 0.003 | 0.809 ± 0.002 |

**Best model: ROS** — highest balanced accuracy (0.860) and F1 score (0.812), with significantly improved recall compared to baseline.

## Tools & Libraries
- Python, Google Colab
- `scikit-learn`, `imbalanced-learn`, `pandas`, `numpy`

## Key Insight
Baseline model has high precision but very low recall (0.590), meaning it **misses 41% of actual cancer cases** — dangerous in a medical context. ROS successfully balances precision-recall tradeoff, making it the most suitable method for this use case.
