import random
from numpy import *

def sample_init(number_species, number_to_sample):
    init_codit_matrix = zeros([100, int(number_species)])
    init_cond_total_matrix = []
    #init_conds = []
    global_min_A2 = 0
    global_max_A2 = 10
    global_min_B2 = 0
    global_max_B2 = 10
    number_segments = 10
    len_segments = (global_max_A2 - global_min_A2)/number_segments
    local_min_A2 = 0
    local_min_B2 = 0

    i = 0
    while local_min_A2 <= 10:
        A2 = random.uniform(local_min_A2, local_min_A2 + len_segments)
        B2 = random.uniform(local_min_B2, local_min_B2 + len_segments)
        #order is: [A, gA, B, gB, A2, B2, A2gB, B2gA]
        init_codit_matrix[i, :] = [0, 1, 0, 1, A2, B2, 0, 0, 0, 0, 0, 0]
        local_min_A2 += len_segments
        i = i+1

        if local_min_A2 == 10:
            local_min_B2 += len_segments
            local_min_A2 = 0
        if local_min_B2 == 10:
            break

    g = 0
    while g <= number_to_sample:
        j = 0
        while j <= 100:
            init_cond_total_matrix.append(init_codit_matrix[j, :])
            j += 1
            if j == 100:
                j = 0
                g += 1
                if g == number_to_sample:
                    break
            if g == number_to_sample:
                break
        if g == number_to_sample:
            break

    return init_cond_total_matrix

if __name__ == "__main__":
    sample_init(10,12)