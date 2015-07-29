from xml.etree import ElementTree as ET

tree = ET.ElementTree(file='input_file.xml')
root = tree.getroot()

lims = []
ics = []

for data in root:
    if data.tag == "epsilon_t":
        epsilons_t = data.text.split()
    elif data.tag == "epsilon_vcl":
        epsilons_vcl = data.text.split()
    elif data.tag == "eps_cl":
        epsilon_cl = data.text.split()
    elif data.tag == "number_of_clusters":
        number_of_clusters = data.text.split()
    elif data.tag == "total_variance":
        total_variance = data.text.split()
    elif data.tag == "cluster_variance":
        cluster_variance = data.text.split()
    elif data.tag == "particles":
        number_particles = data.text
    elif data.tag == "number_to_sample":
        number_to_sample = data.text
    elif data.tag == "initial_conditions_samples":
        initial_conditions_samples = data.text
    elif data.tag == "alpha":
        alpha = data.text
    elif data.tag == "times":
        times = data.text.split()
    elif data.tag == "species_numb_to_fit":
        species_numb_to_fit_lst = data.text.split()
    elif data.tag == "stoch_determ":
        stoch_determ = data.text
    elif data.tag == "parameters":
        for child in data:
            lims.append(child.text.split())
    elif data.tag == "initial_conditions":
        for child in data:
            ics.append(child.text.split())
