scanning_ABC_SMC
================

Example
-------
 
To run the Gardner switch in /examples:
-   Modify run file to your own parameters
-   Run this file
-   To check if the results are all bistable: in plot_population.R, set working directory to the last completed population in results_txt_files/Population_?. Run plot_stabilityChecker_particles funtion.
-   To look at posterior: in plot_population.R, set working directory to the scanning_ABC_SMC directory. Run plot_posterior_distr function. param_names and colnames modify them to the names of your parameters and limits to your prior values.
     plot_posterior_distr(4, param_names, limits, "switch_posterior_b", "Gardner switch posterior distribution")  
     plot_posterior_distr(**number of parameters**, param_names, limits, **"file name"**, **"figure title"**)  

Running a new model
--------------------
-   Add the SBML of you r model to the scanning_ABC_SMC directory
-   In my_abc.py, function *simulate_dataset*: 

    """    Simulate dataset """
#import cudasim.SBMLParser as Parser
#Location of SBML model file
#xmlModel = 'sw_std_dim_deg_sym_sbml.xml'
#name = 'sw_std_dim_deg_sym'
# create CUDA code from SBML model
#Parser.importSBMLCUDA([xmlModel], ['ODE'], ModelName=[name])
    
-   Uncomment the parser and change the name of the xmlModel to the name of your SBML model:

"""    Simulate dataset """
import cudasim.SBMLParser as Parser
Location of SBML model file
xmlModel = '___YOUR MODEL___.xml'
name = 'model'
create CUDA code from SBML model
Parser.importSBMLCUDA([xmlModel], ['ODE'], ModelName=[name])
 
-   Once you have ran this once (and the model.cu file has been created) you can comment this section off again
 
-   In input_file.xml modeify the inputs to what you want. 
input file:

*<epsilon_t>* final distance accepted from desired total variance in order to finish  
*<epsilon_vcl>* final distance accepted from desired cluster variance in order to finish  
*<epsilon_cl>* final distance accepted from desired number of clusters in order to finish  

*<number_of_clusters>* number of clusters desired
*<total_variance>* total variance to aim for
*<cluster_variance>* variance within clusters to aim for

*<particles>* how many need to be accepted to complete the current population
*<number_to_sample>* how many parameter sets to sample
*<initial_conditions_samples>* how many initial conditions to sample  

*<alpha>* Used in selecting the next epsilon from the distance matrix
*<times>* time steps for the simulations
*<species_numb_to_fit>* which species will be used to assess the clusters. There should always be two of them.

*<parameters>* One tag for each parameter. Make sure the first one is "constant 1” as with the copasi convention

*<initial_conditions>* One tag for each species. Make sure they are all "constant x” and only two of them are “uniform x y"
-   In run.sh file change to your own parameters. Keep the following line unchanged: 
    python read_input.py; python my_abc.py
    