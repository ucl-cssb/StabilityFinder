import xml.etree.ElementTree as ET

"""    Read all the data in from input file    """
                   
#Read xml file (input file)
tree = ET.parse('input_file.xml')
root = tree.getroot()

epsilon = root.find('epsilon_t').text
epsilons_t = epsilon.split( )
epsilon_vcl = root.find('epsilon_vcl').text
epsilons_vcl = epsilon_vcl.split( )
epsilons_cl = root.find('eps_cl').text
epsilon_cl = epsilons_cl.split( )


number_of_cluster = root.find('number_of_clusters').text
number_of_clusters = number_of_cluster.split( )
total_variances = root.find('total_variance').text
total_variance = total_variances.split( )
cluster_variances = root.find('cluster_variance').text
cluster_variance = cluster_variances.split( )

time = root.find('times').text
times = time.split( )
species_numb_to_fit = root.find('species_numb_to_fit').text
species_numb_to_fit_lst = species_numb_to_fit.split( )
number_particles = root.find('particles').text
number_to_sample = root.find('number_to_sample').text
initial_conditions_samples = root.find('initial_conditions_samples').text
source = root.find('source').text
#fit = root.find('fit').text
alpha = root.find('alpha').text

parameter1 = root.find('parameter1').text
param_lims1 = parameter1.split()
parameter2 = root.find('ge').text
param_lims2 = parameter2.split()
parameter3 = root.find('rep').text
param_lims3 = parameter3.split()
parameter4 = root.find('rep_r').text
param_lims4 = parameter4.split()
parameter5 = root.find('dim').text
param_lims5 = parameter5.split()
parameter6 = root.find('dim_r').text
param_lims6 = parameter6.split()
parameter7 = root.find('deg').text
param_lims7 = parameter7.split()
parameter8 = root.find('rep_dim').text
param_lims8 = parameter8.split()
parameter9 = root.find('rep_dim_r').text
param_lims9 = parameter9.split()
parameter10 = root.find('deg_sr').text
param_lims10 = parameter10.split()
parameter11 = root.find('deg_dim').text
param_lims11 = parameter11.split()

lims = [param_lims1,param_lims2,param_lims3,param_lims4,param_lims5,param_lims6,param_lims7,param_lims8,param_lims9,param_lims10,param_lims11]

ic1 = root.find('ic1').text
ic_lims1 = ic1.split()
ic2 = root.find('ic2').text
ic_lims2 = ic2.split()
ic3 = root.find('ic3').text
ic_lims3 = ic3.split()
ic4 = root.find('ic4').text
ic_lims4 = ic4.split()
ic5 = root.find('ic5').text
ic_lims5 = ic5.split()
ic6 = root.find('ic6').text
ic_lims6 = ic6.split()
ic7 = root.find('ic7').text
ic_lims7 = ic7.split()
ic8 = root.find('ic8').text
ic_lims8 = ic8.split()
ic9 = root.find('ic9').text
ic_lims9 = ic9.split()
ic10 = root.find('ic10').text
ic_lims10 = ic10.split()
ic11 = root.find('ic11').text
ic_lims11 = ic11.split()
ic12 = root.find('ic12').text
ic_lims12 = ic12.split()

ics = [ic_lims1,ic_lims2,ic_lims3,ic_lims4,ic_lims5,ic_lims6,ic_lims7,ic_lims8,ic_lims9,ic_lims10,ic_lims11,ic_lims12]
