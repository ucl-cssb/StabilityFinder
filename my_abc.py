import read_input
import numpy
from numpy import random
import cudasim
import cudasim.Lsoda as Lsoda
import copy
import time
import operator
import sampl_initi_condit
import clustering
import logging
import matplotlib.pyplot as plt


logging.basicConfig(filename='my_abc_scan.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def central():
    logging.info('ABC started')
    start = time.time()
    number_particles = float(read_input.number_particles)
    number_to_sample = 1000
    logger.debug('number of particles: %s', number_particles)
    logger.debug('number_to_sample: %s', number_to_sample)
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
            timecourseA2, timecourseB2 = simulate_dataset(parameters_sampled, number_to_sample)
            distances_matrix = measure_distance(timecourseA2, timecourseB2, number_to_sample, final_desired_values)
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
                logger.info('accepted_distances: %s', accepted_distances)
                logger.info('param_acc length: %s', len(parameters_accepted))
                break
        fig = plot_steady_states(timecourseA2, timecourseB2, pop_indic, number_particles)
        current_weights_list = particle_weights(parameters_accepted, current_weights_list)
        numpy.savetxt('results_txt_files/Population_'+str(pop_indic+1)+'/data_Population'+str(pop_indic+1)+'.txt', parameters_accepted, delimiter=' ')
        numpy.savetxt('results_txt_files/Population_'+str(pop_indic+1)+'/data_Weights'+str(pop_indic+1)+'.txt', current_weights_list, delimiter=' ')
        pop_indic += 1
              
    while epsilons[0] > epsilons_final[0] or epsilons[1] > epsilons_final[1] or epsilons[2] > epsilons_final[2]:
        timecourseA2 = []
        timecourseB2 = []
        finished = 'false'
        logger.info('population: %s', pop_indic)
        logger.info('population: %s', pop_indic)
        previous_parameters, previous_weights_list, epsilons = prepare_next_pop(parameters_accepted, current_weights_list, final_desired_values, accepted_distances)
        logger.debug('epsilons: %s', epsilons)
        parameters_accepted = []
        accepted_distances = []

        while finished == 'false':
            parameters_sampled, current_sampled_weights = sample_params(previous_parameters, previous_weights_list, number_to_sample)
            perturbed_particles, previous_weights_list = perturb_particles(parameters_sampled, current_sampled_weights, pop_indic)
            timecourseA2, timecourseB2  = simulate_dataset(perturbed_particles, number_to_sample)
            distances_matrix = measure_distance(timecourseA2, timecourseB2, number_to_sample, final_desired_values)
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

        fig = plot_steady_states(timecourseA2, timecourseB2, pop_indic, number_particles)
        current_weights_list = perturbed_particle_weights(parameters_accepted, previous_weights_list, previous_parameters)
        numpy.savetxt('results_txt_files/Population_'+str(pop_indic+1)+'/data_Population'+str(pop_indic+1)+'.txt', parameters_accepted, delimiter=' ')
        numpy.savetxt('results_txt_files/Population_'+str(pop_indic+1)+'/data_Weights'+str(pop_indic+1)+'.txt', current_weights_list, delimiter=' ')
        pop_indic += 1
              
        if epsilons[0] <= epsilons_final[0] and epsilons[1] <= epsilons_final[1] and epsilons[2] <= epsilons_final[2]:
            logger.info('Last population finished')
            fig = plot_steady_states(timecourseA2, timecourseB2, pop_indic, number_particles)
            final_weights = current_weights_list[:]
            final_particles = parameters_accepted[:][:]
            final_timecoursesA2 = timecourseA2[:][:]
            final_timecoursesB2 = timecourseB2[:][:]
            end = time.time()
            logger.debug('TIME: %s', end - start)
            break
    return final_weights, final_particles, final_timecoursesA2, final_timecoursesB2


def prepare_next_pop(parameters_accepted, current_weights_list, final_desired_values, distances_matrix):
    logger.info('Preparing next population')
    logger.debug('distances matrix: %s', distances_matrix)
    distances_matrix.sort(key = operator.itemgetter(0, 1, 2))
    #sort_dist_cl = sorted(distances_matrix, key=itemgetter(0))
    #logger.debug('distances matrix: %s', distances_matrix)
    epsilon_cl_current = distances_matrix[9][0]
    epsilon_t_current = distances_matrix[9][1]
    epsilon_vcl_current = distances_matrix[9][2]

    #alpha = [alpha_cl_current, alpha_t_current, alpha_vcl_current]
    #epsilon_cl_current = abs(alpha_cl_current - final_desired_values[0])
    #epsilon_t_current = abs(alpha_t_current - final_desired_values[1])
    #epsilon_vcl_current = abs(alpha_vcl_current - final_desired_values[2])

    epsilons = [epsilon_cl_current, epsilon_t_current, epsilon_vcl_current]
    logger.debug('alpha: %s', epsilons)

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
    len_list = len(parameters_list)
    len_list_w = len(weights_val_list)
    logger.debug('Number of particles sampled: %s', len_list)
    logger.debug('Number of particle weights: %s', len_list_w)
    return parameters_list, weights_val_list


def simulate_dataset(parameters_sampled, number_to_sample):
       
    init_cond_list = []
    number_species = len(read_input.ics)
    #Here multiply the 'parameters' martix, to repeat each line 100 times, keeping the same order. this is so that the initial conditions and parameters matrices are equal.
    #There are 100 initial conditions per parameter set
    expanded_params_list = []
    logger.info('Expanding parameters list to match initial conditions')
    for i in parameters_sampled:
        j = 0
        while j <= 100:
            expanded_params_list.append(i)
            j += 1
            if j == 100:
                break

    init_cond_list = sampl_initi_condit.sample_init(number_species, number_to_sample)
    len_list = len(expanded_params_list)
    len_list_i = len(init_cond_list)
    logger.debug('Length of expanded parameters list: %s', len_list)
    logger.debug('Length of initial conditions list: %s', len_list_i)
             
    """    Simulate dataset """
    ###############	Create cuda code of model	###########
    #import cudasim.SBMLParser as Parser
    #Location of SBML model file
    #xmlModel = 'sw_std_dim_deg_sym_sbml.xml'
    #name = 'sw_std_dim_deg_sym'
    # create CUDA code from SBML model
    #Parser.importSBMLCUDA([xmlModel], ['ODE'], ModelName=[name])
    #########################################################

    times = read_input.times
    #CUDA SIM BIT
    cudaCode = 'sw_std_dim_deg_sym.cu'
    logger.info('Simulating...')
    modelInstance = Lsoda.Lsoda(times, cudaCode, dt=0.1)
    result = modelInstance.run(expanded_params_list, init_cond_list)
    #[#threads][#beta][#timepoints][#speciesNumber]
    #B2 is species 5 in standard toggle switch
    timecourseB2 = result[:, 0, :, 5]
    timecourseA2 = result[:, 0, :, 4]
    logger.info('finished')
    return timecourseA2, timecourseB2


def measure_distance(timecourseA2, timecourseB2, number_to_sample, final_desired_values ):

    logger.info('Distance module called')
    #Break up the simulated data sets into the parameter sets
    var_t = []
    var_c = []
    cl_c = []
    distances_matrix = []
    g = 0
    f = 0
    f = 0
    while g <= (int(number_to_sample)*100):
        parameter_set = []
        j = 0
        if g == (int(number_to_sample)*100):
            break

        while j <= 100:
            #I only want the last point of the timecourse
            parameter_set.append([timecourseA2[g+j][-1], timecourseB2[g+j][-1]])
            j += 1
            if j == 100:
                #now call the distance function for this group
                cluster_counter, clusters_means, total_variance, median_clust_var = clustering.distance(parameter_set)
                cl_c.append(cluster_counter)
                var_t.append(total_variance)
                var_c.append(median_clust_var)
                distances_matrix.append([abs(cluster_counter - final_desired_values[0]), abs(total_variance - final_desired_values[1]), abs(median_clust_var - final_desired_values[2])])
                g += 100
                f += 1
                break
    logger.debug('Length of distances matrix: %s', len(distances_matrix))
    logger.debug('Length of clusters list: %s', len(cl_c))
    return distances_matrix


def plot_steady_states(timecourseA2, timecourseB2, pop_indic, number_to_sample):
    a2 = []
    b2 = []
    for i in timecourseA2:
        a2.append(i[-1])
    for i in timecourseB2:
        b2.append(i[-1])
    fig, axs = plt.subplots(10, 10, figsize=(30, 20), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.5, wspace=0.1)
    axs = axs.ravel()
    p_set = 0
    i = 0
    while p_set <= int(number_to_sample)-1:
        a2_set = []
        b2_set = []
        tot = 0
        if i == int(number_to_sample)*100:
            break
        if p_set == int(number_to_sample)-1:
            break
        while tot <= 101:
            if i == int(number_to_sample)*100:
                break
            axs[p_set].scatter(a2[i], b2[i], color='blue')
            a2_set.append(a2[i])
            b2_set.append(b2[i])
            i += 1
            tot += 1
            if tot == 100:
            #    numpy.savetxt('results_txt_files/Population_'+str(pop_indic+1)+'/A2_plotting_data_set'+str(p_set)+'.txt', a2_set, delimiter=' ')
            #    numpy.savetxt('results_txt_files/Population_'+str(pop_indic+1)+'/B2_plotting_data_set'+str(p_set)+'.txt', b2_set, delimiter=' ')
                p_set += 1
                break
    #numpy.savetxt('results_txt_files/Population_'+str(pop_indic+1)+'/A2_plotting_data_all.txt', a2, delimiter=' ')
    #numpy.savetxt('results_txt_files/Population_'+str(pop_indic+1)+'/B2_plotting_data_all.txt', b2, delimiter=' ')
    plt.savefig('plot_population_'+str(pop_indic+1)+'.pdf', bbox_inches=0)
    return fig


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
            index_to_delete.append(index)

        if item[1] > epsilons[1] or item[1] != item[1]:
            index_to_delete.append(index)

        if item[2] > epsilons[2] or item[2] != item[2]:
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


def perturb_particles(parameters_sampled,current_weights_list,pop_indic):
    logger.info('Perturbing particles...')
    ##Make a new list which will be used so that you dont confuse them with the sampled parameters
    not_perturbed_particles = copy.deepcopy(parameters_sampled)
    #logger.debug('not_perturbed_particles matrix: %s', not_perturbed_particles)


    perturbed_particles = []
    for particle in not_perturbed_particles:
        part_params = []
        i = 0
        while i < len(particle):
            if i == 0:
                part_params.append(1.0)
                i += 1
            if i > 0:
                minimum = min(param[i] for param in parameters_sampled)
                maximum = max(param[i] for param in parameters_sampled)
                scale = (maximum-minimum)/2
                            
                if particle[i] + scale < float(read_input.lims[i][2]) and particle[i] -scale > float(read_input.lims[i][1]):
                    delta_perturb = random.uniform(low=-scale, high=scale)
                    part_params.append(particle[i] + delta_perturb)
                    i += 1
                elif particle[i] + scale > float(read_input.lims[i][2]):
                    delta_perturb = random.uniform(low=-scale, high= float(read_input.lims[i][2])-particle[i])
                    part_params.append(particle[i] + delta_perturb)
                    i += 1
                elif particle[i] -scale < float(read_input.lims[i][1]):
                    delta_perturb = random.uniform(low=float(read_input.lims[i][1])-particle[i], high=scale)
                    part_params.append(particle[i] + delta_perturb)
                    i += 1
                                   
            if i == len(particle):
                perturbed_particles.append(part_params)
                break
    logger.info('Perturbation finished')
    #logger.debug('perturbed_particles matrix: %s', perturbed_particles)
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
    i = 1
    while i < len(read_input.lims):
        numerator = 0
        minimum_pert = float(read_input.lims[i][1])
        maximum_pert = float(read_input.lims[i][2])
        numer = 1/(maximum_pert-minimum_pert)
        num_tmp.append(numer)
        i += 1
    numerator = reduce(operator.mul, num_tmp, 1)
    """ denominator """
    for particle in range(len(parameters_accepted)):
        #Calculate the probability for each parameter of the particle
        paramet = 1
        params_denom = []
        while paramet < len(read_input.lims):
            denominator_tmp = []
            minimum_prev = min(param[paramet] for param in previous_parameters)
            maximum_prev = max(param[paramet] for param in previous_parameters)
            delta = (maximum_prev-minimum_prev)/2
            for prev_particle in range(len(previous_parameters)):
                denominator_tmp.append(uniform_pdf(parameters_accepted[particle][paramet], previous_parameters[prev_particle][paramet]-delta, previous_parameters[prev_particle][paramet]+delta))
            params_denom.append(sum(denominator_tmp))
            paramet += 1
            if paramet == len(read_input.lims):
                break

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


final_weights, final_particles, final_timecoursesA2, final_timecoursesB2 = central()
numpy.savetxt('Parameter_values_final.txt', final_particles, delimiter=' ')
numpy.savetxt('Parameter_weights_final.txt', final_weights, delimiter=' ')