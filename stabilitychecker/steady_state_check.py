import numpy as np

standard_deviations = []
def ss_check(data):
    for i in data:
       standard_deviations.append(np.std(i, axis=0))
    std_devs = np.median(standard_deviations, axis=0)
    return std_devs


