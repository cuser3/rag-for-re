from math import sqrt
from statsmodels.stats.power import TTestPower

# factors for power analysis
alpha = 0.05
power = 0.8
effect_sizes = [0.2, 0.5, 0.8, 1.3]

obj = TTestPower()
for es in effect_sizes:
    n = obj.solve_power(
        effect_size=es, alpha=alpha, power=power, alternative="two-sided"
    )
    print("Effect Size: " + str(es))
    print("Sample size/Number needed in each group: {:.3f}".format(n))
