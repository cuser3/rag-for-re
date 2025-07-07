import pandas as pd
from scipy.stats import wilcoxon
from scipy.stats import rankdata
from math import sqrt


file_paths = [
    "complete_wilcox.xlsx",
    "singular_wilcox.xlsx",
    "verifiable_wilcox.xlsx",
    "unambiguous_wilcox.xlsx",
    "task_adherence_wilcox.xlsx",
    "context_relevance_wilcox.xlsx",
]
for file_path in file_paths:
    print(file_path)
    data = pd.read_excel(file_path)

    noRAG = data["noRAG"]
    RAG = data["RAG"]

    stat, p_value = wilcoxon(
        noRAG,
        RAG,
        zero_method="pratt",
        correction=False,
        alternative="two-sided",
        nan_policy="omit",
        method="auto",
    )

    differences = noRAG - RAG
    abs_differences = abs(differences)
    ranks = rankdata(abs_differences)  # Assign ranks, averaging tied ranks
    positive_ranks = sum(ranks[differences > 0])
    negative_ranks = sum(ranks[differences < 0])
    print(f"Positive ranks: {positive_ranks}")
    print(f"Negative ranks: {negative_ranks}")

    non_zero_differences = [diff for diff in differences if diff != 0]
    n = len(non_zero_differences)  # Non-zero differences
    print(f"Number of non-zero differences (n): {n}")

    zero_percentage = 1 - (n / len(noRAG.dropna()))
    print(f"Percentage of zeros: {zero_percentage:.2%}")

    expected_w_value = (n * (n + 1)) / 4
    print(f"Expected W value: {expected_w_value}")

    abs_diff_series = pd.Series(abs_differences)
    tie_counts = abs_diff_series.value_counts()
    tied_ranks = tie_counts[tie_counts > 1]
    tie_correction = ((tied_ranks**3 - tied_ranks) / 2).sum()
    print(f"Tie correction value: {tie_correction}")

    std_deviation = sqrt((n * (n + 1) * (2 * n + 1) - tie_correction) / 24)
    print(f"Standard deviation: {std_deviation}")

    test_stat = min(positive_ranks, negative_ranks)
    print(f"Test statistic: {test_stat}")
    print(f"Wilcoxon test statistic: {stat}")

    z_value = (stat - expected_w_value) / std_deviation
    print(f"Z-value: {z_value}")

    effect_size = z_value / sqrt(n)
    print(f"Effect size (r): {effect_size}")
    print(f"P-value: {p_value}")
    print("-" * 50)
