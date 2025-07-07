# Statisitcal Tests

This folder contains data from the user study, aswell as code for running a power analyis and a Wilcoxon signed-rank test.

### Power Analyis
The power analysis was done to determine a minimum sample size for the user study.
To run the power analysis run ``power_analysis.py``.

The script will print out the recommended sample size for different effect sizes accoring to 
[Cohen's Statisitcal Power Analysis](https://journals.sagepub.com/doi/abs/10.1111/1467-8721.ep10768783?casa_token=RKObbhMHJCAAAAAA:ln8lA-3Pd-PqkMkF-d3AWRrLlfmu-gyjnL3Y_bKe7m_auwPTQoeOy-m1N5GB6vftE1pOy0n2dC3b1Q&casa_token=n45LtjsIp3EAAAAA:mI7IboAvLjgl5WbN-aTTuXErtyC5vFwX2N_Y3k0hZg-UPZ2AoMO87CS_XLU89bReptLMNtyRFGusXw).


|Effect Size| Value|
|-----|------|
| Small | 0.2 |
| Medium | 0.5 |
| Large  |  0.8      |
| Very Large  |  1.3    |

### Wilcoxon test
Since the results from the user study provided skewed distributions, we decided to run a non-parametric test instead. Therefore we selected the Wilcoxon signed-rank test.
To run the Wilcoxon test run ``wilcoxon.py``.