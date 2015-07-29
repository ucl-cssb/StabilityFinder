import random
from numpy import *
import read_input


def sample_init(number_to_sample, init_cond_to_sample):

    vals = []
    limits = []
    uniformIndex = []
    p = -1
    for i in read_input.ics:
        p += 1
        if i[0] == 'constant':
            vals.append(float(i[1]))

        if i[0] == 'uniform':
            vals.append(1)
            limits.append(float(i[1]))
            limits.append(float(i[2]))
            uniformIndex.append(p)

    init_codit_matrix = zeros([init_cond_to_sample, len(read_input.ics)])
    init_cond_total_matrix = []
    global_min_1 = limits[0]
    global_max_1 = limits[1]
    global_min_2 = limits[2]
    global_max_2 = limits[3]
    number_segments = sqrt(init_cond_to_sample)
    len_segments = (global_max_1 - global_min_1)/number_segments
    local_min_1 = limits[0]
    local_min_2 = limits[2]

    i = 0
    while local_min_1 <= global_max_1:


        A2 = random.uniform(local_min_1, local_min_1 + len_segments)
        B2 = random.uniform(local_min_2, local_min_2 + len_segments)

        vals[uniformIndex[0]] = A2
        vals[uniformIndex[1]] = B2

        init_codit_matrix[i, :] = vals
        local_min_1 += len_segments
        i = i+1

        if local_min_1 == global_max_1:
            local_min_2 += len_segments
            local_min_1 = global_min_1
        if local_min_2 == global_max_2:
            break

    g = 0
    while g <= number_to_sample:
        j = 0
        while j <= init_cond_to_sample:
            init_cond_total_matrix.append(init_codit_matrix[j, :])
            j += 1
            if j == init_cond_to_sample:
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
    sample_init(12, 100)