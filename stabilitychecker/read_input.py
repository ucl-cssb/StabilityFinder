from xml.etree import ElementTree

document = ElementTree.parse('input_file.xml')

epsilon_t = document.find('epsilon_t')
epsilons_t = [epsilon_t.find('start').text, epsilon_t.find('end').text]
epsilon_vcl = document.find('epsilon_vcl')
epsilons_vcl = [epsilon_vcl.find('start').text, epsilon_vcl.find('end').text]
epsilons_cl = document.find('eps_cl')
epsilon_cl = [epsilons_cl.find('start').text, epsilons_cl.find('end').text]
number_of_clust = document.find('number_of_clusters')
number_of_clusters = [number_of_clust.find('start').text, number_of_clust.find('end').text]
total_var = document.find('total_variance')
total_variance = [total_var.find('start').text, total_var.find('end').text]
cluster_var = document.find('cluster_variance')
cluster_variance = [cluster_var.find('start').text, cluster_var.find('end').text]
numb_part = document.find('particles')
number_particles = numb_part.text
numb_to_samp = document.find('number_to_sample')
number_to_sample = numb_to_samp.text
initial_cond_samp = document.find('initial_conditions_samples')
initial_conditions_samples = initial_cond_samp.text
a = document.find('alpha')
alpha = a.text
t = document.find('times')
times = t.text.split()
spec_numb_fit = document.find('species_numb_to_fit')
species_numb_to_fit_lst = spec_numb_fit.text.split()
stoch_det = document.find('stoch_determ')
stoch_determ = stoch_det.text

lims = []
ics = []

for item in document.find('parameters').getchildren():
   lims.append([item.find('distribution').text, item.find('start').text, item.find('end').text])
for item in document.find('initial_conditions').getchildren():
   ics.append([item.find('distribution').text, item.find('start').text, item.find('end').text])

