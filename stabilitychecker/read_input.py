from xml.etree import ElementTree

def inp(filename):


    document = ElementTree.parse(filename)

    epsilon_t = document.find('epsilon_t')
    epsilons_t = epsilon_t.find('end').text
    epsilon_vcl = document.find('epsilon_vcl')
    epsilons_vcl = epsilon_vcl.find('end').text
    epsilons_cl = document.find('eps_cl')
    epsilon_cl =  epsilons_cl.find('end').text
    #epsilons = [float(epsilon_cl[0]), float(epsilons_t[0]), float(epsilons_vcl[0])]
    epsilons_final = [float(epsilon_cl), float(epsilons_t), float(epsilons_vcl)]


    ss = document.find('steady_state')
    steady_state_std = ss.find('standard_dev').text
    cluster_mean = float(ss.find('cluster_mean').text)

    number_of_clust = document.find('number_of_clusters')
    number_of_clusters = number_of_clust.find('end').text
    total_var = document.find('total_variance')
    total_variance = total_var.find('end').text
    cluster_var = document.find('cluster_variance')
    cluster_variance = cluster_var.find('end').text
    final_desired_values = [float(number_of_clusters), float(total_variance), float(cluster_variance)]

    numb_part = document.find('particles')
    number_particles = float(numb_part.text)
    numb_to_samp = document.find('number_to_sample')
    number_to_sample = float(numb_to_samp.text)
    initial_cond_samp = document.find('initial_conditions_samples')
    initial_conditions_samples = int(initial_cond_samp.text)
    a = document.find('alpha')
    alpha = float(a.text)

    dta = document.find('dt')
    dt = float(dta.text)
    delt = document.find('det_clust_delta')
    det_clust_delta = float(delt.text)
    cf = document.find('kmeans_cutoff')
    kmeans_cutoff = float(cf.text)

    ts = document.find('times_start')
    te = document.find('times_end')
    ttd = document.find('times_twidth')

    st = float(ts.text)
    end = float(te.text)
    twidth = float(ttd.text)

    spec_numb_fit = document.find('species_numb_to_fit')
    species_numb_to_fit_lst = spec_numb_fit.text.split()
    stoch_det = document.find('stoch_determ')
    stoch_determ = stoch_det.text

    clust  = document.find('clustering')
    clustering = clust.text
    mod_file = document.find('model_file')
    model_file = mod_file.text
    mod = document.find('sbml_name')
    sbml_name = mod.text

    lims = []
    ics = []
    ignore_param = document.find('cell_volume_first_param')
    cell_volume_first_param = ignore_param.text
    for item in document.find('parameters').getchildren():
        lims.append([item.find('distribution').text, item.find('start').text, item.find('end').text])
    for item in document.find('initial_conditions').getchildren():
        ics.append([item.find('distribution').text, item.find('start').text, item.find('end').text])

    return epsilons_final, final_desired_values, steady_state_std, cluster_mean, number_particles,\
           number_to_sample, initial_conditions_samples, alpha, st, end, twidth, species_numb_to_fit_lst, stoch_determ, \
           clustering, model_file, sbml_name, lims, ics, dt, det_clust_delta, kmeans_cutoff, cell_volume_first_param
