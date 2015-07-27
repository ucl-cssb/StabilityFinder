import read_input
import numpy
from numpy import random
import copy
import time
import operator
import sampl_initi_condit
import deterministic_clustering
import logging
import steady_state_check
import matplotlib.pyplot as plt
import math
import sys
import os
sys.path.append("/home/ucbtle1/cuda-sim-code")
import cudasim
import cudasim.Lsoda as Lsoda
import cudasim.EulerMaruyama as EulerMaruyama
import cudasim.Gillespie as Gillespie


logging.basicConfig(filename='my_abc_scan.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def central():
    logging.info('ABC started')
    for i in range(1, 10):
        if os.path.exists('results_txt_files_'+str(i)):
            i += 1
        else:
            os.makedirs('results_txt_files_'+str(i))
            logger.info('Made results directory')
            break
    results_path = 'results_txt_files_'+str(i)
    start = time.time()
    number_particles = int(read_input.number_particles)
    number_to_sample = int(read_input.number_to_sample)
    init_cond_to_sample = int(read_input.initial_conditions_samples)
    alpha = math.ceil(float(read_input.alpha)*number_particles)
    logger.debug('alpha: %s', alpha)
    species_numb_to_fit = read_input.species_numb_to_fit_lst
    logger.debug('number of particles: %s', number_particles)
    logger.debug('number_to_sample: %s', number_to_sample)
    logger.debug('species_numb_to_fit: %s', species_numb_to_fit)

    pop_indic = 0
    current_weights_list = []
    parameters_accepted = []
    accepted_distances = []
    epsilons_final = [float(read_input.epsilon_cl[1]), float(read_input.epsilons_t[1]), float(read_input.epsilons_vcl[1])]

    final_number_clusters = float(read_input.number_of_clusters[1])
    final_total_variance = float(read_input.total_variance[1])
    final_cluster_variance = float(read_input.cluster_variance[1])
    final_desired_values = [final_number_clusters, final_total_variance, final_cluster_variance]

    if pop_indic == 0:
        finished = 'false'
        logger.info('population: %s', pop_indic)
        epsilon_t_current = float(read_input.epsilons_t[0])
        epsilon_vcl_current = float(read_input.epsilons_vcl[0])
        epsilon_cl_current = float(read_input.epsilon_cl[0])
        epsilons = [epsilon_cl_current, epsilon_t_current, epsilon_vcl_current]

        logger.debug('epsilon_t_current: %s', epsilon_t_current)
        logger.debug('epsilon_vcl_current: %s', epsilon_vcl_current)
        logger.debug('epsilon_cl_current: %s', epsilon_cl_current)


        while finished == 'false':
            parameters_sampled = sample_priors(number_to_sample)

            cudasim_result = simulate_dataset(parameters_sampled, number_to_sample, init_cond_to_sample)
            distances_matrix = measure_distance(cudasim_result, number_to_sample, final_desired_values, init_cond_to_sample, species_numb_to_fit)
            parameters_sampled, distances_matrix = accept_reject_params(distances_matrix, parameters_sampled, epsilons)
            for i in parameters_sampled:
                parameters_accepted.append(i)
            for i in distances_matrix:
                accepted_distances.append(i)

            logger.debug('Number of accepted distances matrix: %s', len(accepted_distances))
            if len(parameters_accepted) >= float(read_input.number_particles):
                parameters_accepted = parameters_accepted[0:int(read_input.number_particles)]
                accepted_distances = accepted_distances[0:int(read_input.number_particles)]
                finished = 'true'
                logger.info('Reached number of particles ')
            elif len(parameters_accepted) < float(read_input.number_particles):
                finished = 'false'
                logger.info('Not reached number of particles, sampling again')
                logger.info('Total number of particles accepted: %s', len(parameters_accepted))
            if finished == 'true':
                logger.info('accepted_distances: %s', len(accepted_distances))
                logger.info('param_acc length: %s', len(parameters_accepted))
                break

        pop_fold_res_path = 'Population_'+str(pop_indic+1)
        os.makedirs(str(results_path)+'/'+str(pop_fold_res_path))
        fig = plot_steady_states(cudasim_result, pop_indic, number_particles, init_cond_to_sample, species_numb_to_fit,results_path,pop_fold_res_path)
        current_weights_list = particle_weights(parameters_accepted, current_weights_list)
        numpy.savetxt(str(results_path)+'/'+str(pop_fold_res_path)+'/data_Population'+str(pop_indic+1)+'.txt', parameters_accepted, delimiter=' ')
        numpy.savetxt(str(results_path)+'/'+str(pop_fold_res_path)+'/data_Weights'+str(pop_indic+1)+'.txt', current_weights_list, delimiter=' ')
        pop_indic += 1
              
    while epsilons[0] > epsilons_final[0] or epsilons[1] > epsilons_final[1] or epsilons[2] > epsilons_final[2]:

        finished = 'false'
        logger.info('population: %s', pop_indic)
        pop_fold_res_path = 'Population_'+str(pop_indic+1)
        os.makedirs(str(results_path)+'/'+str(pop_fold_res_path))
        previous_parameters, previous_weights_list, epsilons = prepare_next_pop(parameters_accepted, current_weights_list, alpha, accepted_distances)
        logger.debug('epsilons: %s', epsilons)
        parameters_accepted = []
        accepted_distances = []

        while finished == 'false':
            parameters_sampled, current_sampled_weights = sample_params(previous_parameters, previous_weights_list, number_to_sample)
            perturbed_particles, previous_weights_list = perturb_particles(parameters_sampled, current_sampled_weights, pop_indic)
            cudasim_result = simulate_dataset(perturbed_particles, number_to_sample, init_cond_to_sample)
            distances_matrix = measure_distance(cudasim_result, number_to_sample, final_desired_values, init_cond_to_sample, species_numb_to_fit)
            parameters_sampled, distances_matrix = accept_reject_params(distances_matrix, perturbed_particles, epsilons)
            # Append the accepted ones to a matrix which will be built up until you reach the number of particles you want
            for i in parameters_sampled:
                parameters_accepted.append(i)
            for i in distances_matrix:
                accepted_distances.append(i)
            logger.debug('Number of accepted distances matrix: %s', len(accepted_distances))

            if len(parameters_accepted) >= float(read_input.number_particles):
                parameters_accepted = parameters_accepted[0:int(read_input.number_particles)]
                accepted_distances = accepted_distances[0:int(read_input.number_particles)]
                finished = 'true'
                logger.info('Reached number of particles ')
            elif len(parameters_accepted) < float(read_input.number_particles):
                finished = 'false'
                logger.info('Not reached number of particles, sampling again')
            if finished == 'true':
                break

        fig = plot_steady_states(cudasim_result, pop_indic, number_particles, init_cond_to_sample, species_numb_to_fit,results_path,pop_fold_res_path)
        current_weights_list = perturbed_particle_weights(parameters_accepted, previous_weights_list, previous_parameters)
        numpy.savetxt(str(results_path)+'/'+str(pop_fold_res_path)+'/data_Population'+str(pop_indic+1)+'.txt', parameters_accepted, delimiter=' ')
        numpy.savetxt(str(results_path)+'/'+str(pop_fold_res_path)+'/data_Weights'+str(pop_indic+1)+'.txt', current_weights_list, delimiter=' ')
        pop_indic += 1
              
        if epsilons[0] <= epsilons_final[0] and epsilons[1] <= epsilons_final[1] and epsilons[2] <= epsilons_final[2]:
            logger.info('Last population finished')
            fig = plot_steady_states(cudasim_result, pop_indic, number_particles, init_cond_to_sample, species_numb_to_fit, results_path,pop_fold_res_path)
            final_weights = current_weights_list[:]
            final_particles = parameters_accepted[:][:]
            final_timecourse1 = cudasim_result[:, 0, :, int(species_numb_to_fit[0])-1]
            final_timecourse2 = cudasim_result[:, 0, :, int(species_numb_to_fit[1])-1]
            end = time.time()
            logger.debug('TIME: %s', end - start)
            break
    return final_weights, final_particles, final_timecourse1, final_timecourse2, pop_fold_res_path,results_path


def prepare_next_pop(parameters_accepted, current_weights_list, alpha, distances_matrix):
    logger.info('Preparing next population')
    logger.debug('length of distances matrix: %s', len(distances_matrix))
    distances_matrix.sort(key=operator.itemgetter(0, 1, 2))
    epsilon_cl_current = distances_matrix[int(alpha)][0]
    epsilon_t_current = distances_matrix[int(alpha)][1]
    #epsilon_t_current = round(epsilon_t_current, 4)
    epsilon_vcl_current = distances_matrix[int(alpha)][2]
    #epsilon_vcl_current = round(epsilon_vcl_current, 4)
    epsilons = [epsilon_cl_current, epsilon_t_current, epsilon_vcl_current]
    logger.debug('epsilons: %s', epsilons)
    return parameters_accepted, current_weights_list, epsilons


def sample_priors(number_to_sample):
    logger.info('sampling priors')
    parameters_list = []
    partic_indic = 0
    while partic_indic <= number_to_sample:
        params = []
        for i in read_input.lims:
            if i[0] == 'constant':
                params.append(1.0)
            elif i[0] == 'uniform':
                params.append(random.uniform(low=i[1], high=i[2]))
        parameters_list.append(params)
        partic_indic += 1
        if partic_indic == number_to_sample:
            break
    len_list = len(parameters_list)
    logger.debug('Number of parameter sets sampled: %s', len_list)
    return parameters_list


def sample_params(parameters_accepted, current_weights_list, number_to_sample):
    logger.info('sampling particles from previous population')


    def choose_sample(current_weights_list):
        #Bernouli trials
        #Choose a random number between 0 and 1
        n = random.random_sample()
        for i in range(0, len(current_weights_list)):
            #|-----|--|-------|---|-----|
            #If you are in the current segment, great!
            if n < current_weights_list[i]:
                break
            #if not, go to the next segment in remove the - you already counter
            n = n - current_weights_list[i]
            #Great, you've got your -!
        return i
       
    weights_val_list = []
    partic_indic = 0
    weight_index_list = []
    parameters_list = []
    while partic_indic < number_to_sample:
        sample = choose_sample(current_weights_list)
        weight_index_list.append(sample)
        partic_indic += 1
        if partic_indic == number_to_sample:
            break
    for index in weight_index_list:
        weight_val = current_weights_list[index]
        weights_val_list.append(weight_val)
        param = parameters_accepted[index]
        parameters_list.append(param)
    logger.debug('Number of particles sampled: %s', len(parameters_list))
    logger.debug('Number of particle weights: %s', len(weights_val_list))
    return parameters_list, weights_val_list


def simulate_dataset(parameters_sampled, number_to_sample, init_cond_to_sample):
       
    init_cond_list = []
    number_species = len(read_input.ics)
    #Here multiply the 'parameters' martix, to repeat each line 100 times, keeping the same order. this is so that the initial conditions and parameters matrices are equal.
    #There are 100 initial conditions per parameter set
    expanded_params_list = []
    logger.info('Expanding parameters list to match initial conditions')
    logger.debug('Length of parameters list: %s', len(parameters_sampled))
    for i in parameters_sampled:
        for j in range(0, init_cond_to_sample):
            expanded_params_list.append(i)

    init_cond_list = sampl_initi_condit.sample_init(number_to_sample, init_cond_to_sample)
    #logger.debug('initial conditions:%s', init_cond_list)
    logger.debug('Length of expanded parameters list: %s', len(expanded_params_list))
    logger.debug('Length of initial conditions list: %s', len(init_cond_list))

    """    Simulate dataset """
    ###############	Create cuda code of model	###########
    #import cudasim.SBMLParser as Parser
    #Location of SBML model file
    #xmlModel = 'sw_std_dim_deg_sym_sbml.xml'
    #name = 'model'
    # create CUDA code from SBML model
    #Parser.importSBMLCUDA([xmlModel], ['ODE'], ModelName=[name])
    #########################################################

    times = read_input.times
    #CUDA SIM BIT
    cudaCode = 'model.cu'
    logger.info('Simulating...')
    modelInstance = Lsoda.Lsoda(times, cudaCode, dt=0.1)
    result = modelInstance.run(expanded_params_list, init_cond_list)
    #logger.debug('simulation result: %s', result)
    #[#threads][#beta][#timepoints][#speciesNumber]
    logger.info('finished')
    return result

def measure_distance(cudasim_result, number_to_sample, final_desired_values, init_cond_to_sample, species_numb_to_fit):

    logger.info('Distance module called')
    distances_matrix = []
    for i in range(0, int(number_to_sample)):
        range_start = i*int(init_cond_to_sample)
        range_end = i*int(init_cond_to_sample) + int(init_cond_to_sample) - 1
        #[#threads][#beta][#timepoints][#speciesNumber]
        set_result = cudasim_result[range_start:range_end, 0, -1, int(species_numb_to_fit[0])-1:int(species_numb_to_fit[1])]
        ss_res_set = cudasim_result[range_start:range_end, 0, -10:, int(species_numb_to_fit[0])-1:int(species_numb_to_fit[1])]
        std_devs = steady_state_check.ss_check(ss_res_set)
        cluster_counter, clusters_means, total_variance, median_clust_var = deterministic_clustering.distance(set_result)
        #logger.debug('cluster counter: %s', cluster_counter)
        distances_matrix.append([abs(cluster_counter - final_desired_values[0]), abs(total_variance - final_desired_values[1]), abs(median_clust_var - final_desired_values[2]), std_devs[0], std_devs[1]])

    logger.info('Distance finished')
    logger.debug('Distance matrix: %s', distances_matrix)
    return distances_matrix


def accept_reject_params(distances_matrix, parameters_sampled, epsilons):
    logger.info('Accepting or rejecting particles')
    #Reject the priors>e.
    index_to_delete = []
    #new_list = sorted(distances_matrix, key=itemgetter(0))
    #logger.debug('distances matrix: %s', distances_matrix)
    for index, item in enumerate(distances_matrix):
        #epsilon_cl_current is the distance of current alpha to desired behaviour!
        #item!=item is a check in case the value is nan
        if item[0] > epsilons[0] or item[0] != item[0]:
            logger.debug('cluster counter failed')
            index_to_delete.append(index)

        if item[1] > epsilons[1] or item[1] != item[1]:
            index_to_delete.append(index)
            logger.debug('total variance failed')

        if item[2] > epsilons[2] or item[2] != item[2]:
            index_to_delete.append(index)
            logger.debug('cluster variance failed')

        if item[3] > 0.000001 or item[4] > 0.000001:
            logger.debug('steady state check failed')
            index_to_delete.append(index)

    #get the unique values by converting the list to a set
    index_to_delete = set(index_to_delete)
    index_to_delete_l = list(index_to_delete)
    sorted_index_delete = sorted(index_to_delete_l)
    logger.debug('indexes to delete: %s', sorted_index_delete)
    if len(sorted_index_delete) > 0:
        for index in reversed(sorted_index_delete):
            del distances_matrix[index]
            del parameters_sampled[index]
    return parameters_sampled, distances_matrix

def plot_steady_states(cudasim_result, pop_indic, number_to_sample, init_cond_to_sample, species_numb_to_fit, results_path,pop_fold_res_path ):
    #[#threads][#beta][#timepoints][#speciesNumber]
    logger.info('saving result to file')
    #fig, axs = plt.subplots(10, 10, figsize=(30, 20), facecolor='w', edgecolor='k')
    #fig.subplots_adjust(hspace=.5, wspace=0.1)
    #axs = axs.ravel()
    p=1
    for i in range(0, int(number_to_sample)):
        range_start = i*int(init_cond_to_sample)
        range_end = i*int(init_cond_to_sample) + int(init_cond_to_sample)
        set_result = cudasim_result[range_start:range_end, 0, -1, int(species_numb_to_fit[0])-1:int(species_numb_to_fit[1])]
        #numpy.savetxt('set_result'+str(i)+'.txt', set_result, delimiter=' ')
        numpy.savetxt(str(results_path)+'/'+str(pop_fold_res_path)+'/set_result'+str(i)+'.txt', set_result, delimiter=' ')
    #    for j in set_result:
    #        axs[i].scatter(j[0], j[1], color='blue')
    #plt.savefig('plot_population_'+str(pop_indic+1)+'.pdf', bbox_inches=0)
    logger.info('saving done')
    return p


def particle_weights(parameters_sampled, weights_list):
    logger.info('Calculating weights')

    """    Weights of particles for t=0   """
    for i in range(len(parameters_sampled)):
        weights_list.append(1)

    """    Normalise the weights    """
    n = sum(weights_list)
    for i in range(len(weights_list)):
        weights_list[i] = float(weights_list[i])/float(n)
    return weights_list


def perturb_particles(parameters_sampled,current_weights_list, pop_indic):
    logger.info('Perturbing particles...')
    ##Make a new list which will be used so that you dont confuse them with the sampled parameters
    not_perturbed_particles = copy.deepcopy(parameters_sampled)
    perturbed_particles = []
    for particle in not_perturbed_particles:
        part_params = []

        for i in range(0, len(particle)):
            if i == 0:
                part_params.append(1.0)
            if i > 0:
                minimum = min(param[i] for param in parameters_sampled)
                maximum = max(param[i] for param in parameters_sampled)
                scale = (maximum-minimum)/2

                if particle[i] + scale < float(read_input.lims[i][2]) and particle[i] - scale > float(read_input.lims[i][1]):
                    delta_perturb = random.uniform(low=-scale, high=scale)
                    part_params.append(particle[i] + delta_perturb)
                    i += 1
                elif particle[i] + scale > float(read_input.lims[i][2]):
                    delta_perturb = random.uniform(low=-scale, high= float(read_input.lims[i][2])-particle[i])
                    part_params.append(particle[i] + delta_perturb)
                    i += 1
                elif particle[i] - scale < float(read_input.lims[i][1]):
                    delta_perturb = random.uniform(low=float(read_input.lims[i][1])-particle[i], high=scale)
                    part_params.append(particle[i] + delta_perturb)
                    i += 1

            if i == len(particle):
                perturbed_particles.append(part_params)
    logger.info('Perturbation finished')
    return perturbed_particles, current_weights_list


def uniform_pdf(x, a, b):
    j = 0
    if x <= b and x > a:
        j = b-a
        return 1/j
    else:
        return 0


def perturbed_particle_weights(parameters_accepted, prev_weights_list, previous_parameters):
    logger.info('Calculating weights')
    current_weights_list = []
    """ numerator """
    num_tmp = []
    for i in range(1, len(read_input.lims)):
        numerator = 0
        minimum_pert = float(read_input.lims[i][1])
        maximum_pert = float(read_input.lims[i][2])
        numer = 1/(maximum_pert-minimum_pert)
        num_tmp.append(numer)
    numerator = reduce(operator.mul, num_tmp, 1)
    """ denominator """
    for particle in range(len(parameters_accepted)):
        #Calculate the probability for each parameter of the particle
        params_denom = []
        for paramet in range(1, len(read_input.lims)):
            denominator_tmp = []
            minimum_prev = min(param[paramet] for param in previous_parameters)
            maximum_prev = max(param[paramet] for param in previous_parameters)
            delta = (maximum_prev-minimum_prev)/2
            for prev_particle in range(len(previous_parameters)):
                denominator_tmp.append(uniform_pdf(parameters_accepted[particle][paramet], previous_parameters[prev_particle][paramet]-delta, previous_parameters[prev_particle][paramet]+delta))
            params_denom.append(sum(denominator_tmp))

        #reduce calculates the cumulative sum from left to to right.
        #operator.mul multiplies
        #this is equivalent to for i in list, p*=i
        particle_denominator = reduce(operator.mul, params_denom, 1)
        current_weights_list.append(numerator/particle_denominator)
              
    """    Normalise the weights    """
    n = sum(current_weights_list)
    for i in range(len(current_weights_list)):
        current_weights_list[i] = float(current_weights_list[i])/float(n)
    logger.debug('perturbed_particles weights matrix: %s', current_weights_list)
    return current_weights_list

final_weights, final_particles, final_timecourses1, final_timecourses2, pop_fold_res_path,results_path = central()
final_path_v = str(results_path)+'/'+str(pop_fold_res_path)+'/Parameter_values_final.txt'
final_path_w = str(results_path)+'/'+str(pop_fold_res_path)+'/Parameter_weights_final.txt'
numpy.savetxt(final_path_v, final_particles, delimiter=' ')
numpy.savetxt(final_path_w, final_weights, delimiter=' ')