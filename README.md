

Installation
-------------

StabilityFinder has been developed to work on a Linux operating system, with GPU support.
The following dependencies are essential for the successful implementation of the package, and must be downloaded and installed:
  
Python 2

https://www.python.org/ftp/python/2.7.12/Python-2.7.12rc1.tgz

R 

https://www.r-project.org/
  
Python packages:

* numpy: https://sourceforge.net/projects/numpy/files/latest/download?source=files
* pycuda: https://pypi.python.org/packages/e8/3d/4b6b622d8a22cace237abad661a85b289c6f0803ccfa3d2386103307713c/pycuda-2016.1.2.tar.gz

R packages:
* ggplot2

Other packages:

 * cuda-sim: https://sourceforge.net/projects/cuda-sim/
 * libsbml (for SBML interface): http://downloads.sourceforge.net/project/sbml/libsbml/4.0.1/libsbml-4.0.1-src.zip
 
Once these have been downloaded and successfully installed, the following commands should be used to install StabilityFinder:
* git clone https://github.com/ucl-cssb/StabilityFinder.git 
* cd StabilityFinder
* python setup.py install --home=\<dir\>

where \<dir\> is the directory where you want to install to. (Note that you can omit the --home=\<dir\> argument and the package will be installed where Python is installed).
This will copy the module StabilityFinder into the

\<dir\>/lib

\<dir\>/bin

directories. 

Add the script directory to the path and the lib directory to the Python path (must be done in each session or added to .bashrc file)

export PATH=\<dir\>/bin:$PATH 

export PYTHONPATH=\<dir\>/lib:$PYTHONPATH
	
The user is now ready to run the examples provided with the package. 
	
<!---(An exe=\<dir\> variable should also be added
to the run.sh script pointing the script to the right package installation. Alternatively the path to the
module can be added to the top of the run.sh file without the need for installation. The path to the
cuda sim module must be added to the top of the run.sh file.)

$ export PATH=\<dir\>:$PATH--->


Using the package
-------------------
###The working directory

The working directory must contain the following files. Each one is described in detail in the section
following.
* input file.xml: The user input file
* model file: This can take two formats; a cuda file named model.cu; or an SBML model which can have any name, as long as this is provided in the input_file.xml.
cuda-sim will create the .cu file using this SBML model.


###The input file

The input.xml file contains all the necessary information that is not contained in the model itself. This
file must follow the specific format showed in the 'Examples' folder. No whitespaces are to be included
between the tags.

####Required arguments

* **epsilons** The acceptable distance from the desired value.
  * **epsilon t** Total variance in the data
    * **end** The final cut-off value for the distance from the desired values.
  * **epsilon_vcl** Within-cluster variance
    * **end** The final cut-off value for the distance from the desired values.
  * **eps_cl** Number of clusters
    * **end** The final cut-off value for the distance from the desired values.
  
* Desired final values The desired values the algorithm will converge to.
  * **steady_state** 
    * **standard_dev** The standard deviation of the last 10 time points that will determine of the system is in steady state
    * **cluster_mean** The minimum value for the mean of the clusters
  * **number of clusters** The number of clusters in the data,describing its stability.
  * **total variance** The total variance in the data.
  * **cluster variance** The within-cluster variance in the data.
* **particles** The number of accepted parameter sets required to proceed to the next iteration.
* **number to sample** Number of parameter sets to sample from for iteration. This needs to be greater or equal to the particles.
* **initial conditions samples** Number of initial condition sets to sample for each parameter set at each iteration. The value has to have an integer sqare root.
* **alpha** A value that describes the step size for the incremental reduction of the epsilon at each iteration. This should be set to a small value (a = 0.1) to speed up the algorithm, or a larger value to take smaller steps. 
* **dt** the time step to be used for the simulaions. If the system is deterministic, this should be set to -1.
* **det_clust_delta** 
* **kmeans_cutoff**
* **times** 
  * **times_start** The first timepoint
  * **times_end** The last timepoint
  * **times_twidth** The step size of the time structure
* **species numb to fit** Which two species numbers to be fit by the algorithm. The should be whitespace delimited.
* **stoch determ** The type of simulation to be used. The two options are stochastic or deterministic.
* **clustering** The clustering algorithm to be used. The two options are gapstatistic or det.
* **cell_volume_first_param** True or False. This dictates whether the first parameter is to be ignored.
* **model file** Whether the model provided is in SBML or cuda format. The two options are sbml and cuda.
* **sbml name** The name of the SBML file that will be provided. 

* **parameters** The parameters included in the model.
  * **item** One item corresponds to one parameter. The number and order of the items must strictly correspond to those of the parameters in the model provided.
    * **name** The name of the parameter. Note that the parameters are read by order and not name from the model. The name provided here is used in the plotting module.
    * **distribution** From what distribution to sample values from. The two options are constant and uniform.
    * **start** The lower limit of the distribution to sample from.
    * **end** The upper limit of the distribution to sample from. Note that if the distribution is constant this value is not read.
* **initial conditions** The species included in the model.
  * **item** One item corresponds to one species. The number and order of the items must strictly correspond to those of the species in the model provided.
    * **name** The name of the species. Note that the species are read by order and not name from the model.
    * **distribution** From what distribution to sample values from. The two options are constant and uniform. Strictly two species must be set to uniform and the rest constant.
    * **start** The lower limit of the distribution to sample from.
    * **end** The upper limit of the distribution to sample from. Note that if the distribution is constant this value is not read.

####Output

The outputs from running StabilityFinder are saved in the results_txt_files folder. This folder
contains two files and one folder per population. The two files are the following:
* **Parameter_values_final.txt** Each line contains the values of the parameters in the order set in
the input_file.xml file and each line corresponds to an accepted particle in the final accepted
population.
* **Parameter_weights_final.txt** Contains the weights of each particle (parameter set) in the final ac-
cepted population.
* **Initial_conditions_final.txt** Contains the initial condition values sampled for the two proteins of interest.
Each population folder contains the following files:
* **data_PopulationN.txt** Each line contains the values of the parameters in the order set in the
input_file.xml file and each line corresponds to an accepted particle.
* **data_WeightsN.txt** Contains the weights of each particle (parameter set).
* One **set_resultXX** per parameter set. Each line contains the steady state value of each species, in order specified in the input_file.xml file, species numb to fit. Each line corresponds to one initial condition set.
The plot of the posterior is saved in the posterior.pdf file in the working directory. The plot is interpreted as follows:
* The marginal distributions of each parameter are found on the diagonal
* The pairwise joint distributions are found along the sides. The location of each pairwise joint distribution is determined by which pair of parameters are compared.

Running StabilityFinder
------------------------


Examples
---------
####The simple genetic toggle switch using ODEs
#####Setting up

**Model**
This example does not use an SBML model but instead provides StabilityFinder with the cuda file
directly. This is the model.cu file.

**Input file**
The input file is set up as shown in the examples folder. As this model only contains two species, u and v, these are selected for the fit as well as initial condition scan.


#####Running the example
Navigate to the folder containing the example, which is under StabilityFinder/examples/Gardner/Deterministic/

The working directory must contain the following:
* The model.cu file
* The input.xml file

The algorithm is initiated by typing:

stabilityFinder -i input_file.xml -o results -l log.log

The progress of the algorithm can be followed in the log.log file. Once StabilityFinder is finished the posterior and the phase plots of the populations can be visualised by running the R code supplied. 
 
This is done by navigating to the examples/ folder where the R scripts are placed and typing the following:

Phase plots:

Rscript plot_phase_space.R Gardner_switch/Deterministic/res/Population_2/

Posterior distribution plot:

Rscript plot_posterior.R Gardner_switch/Deterministic/input_file.xml Gardner_switch/Deterministic/res/ TRUE posterior.pdf
 
The resulting phase plot is in the Gardner_switch/Deterministic/res/Population_2/ directory and the posterior distribution in the Gardner_switch/Deterministic/res/ directory. The arguments required for the R scripts are described in more detail below: 
 
#####Results

######Plotting the results
The results can be visualised by running the R code supplied in the examples folder. To plot the phase plot type the following command in the directory where the plot_phase_space.R file is:


$Rscript plot_phase_space.R \<last_population_directory_file_path\>


* **\<last_population_directory_file_path\>** the file path to the folder of the last population

The resulting phase plots are located in the \<last_population_directory_file_path\> folder. 

To plot the posterior distribution, type the following command in the directory where the plot_posterior.R file is: 


$Rscript plot_posterior.R \<input_file_path\> \<file_path_to_results_directory\> ignore_first_parameter outfile 



* **\<input_file_path\>** the file path and the name of the input file
* **\<file_path_to_results_directory\>** the file path to the results directory
* **ignore_first_parameter** TRUE/FALSE value. If TRUE, the first parameter is ignored 
* **outfile** the name of the posterior output file

The resulting posterior distribution plot is located in the \<file_path_to_results_directory\> folder, unless specified otherwise in the outfile argument. 

For the posterior distribution figure, the plots on the diagonal represent the marginal distributions for the values of each parameter that were found in the
final population, thus the ones that can produce bistable behaviour. The pairwise joint distributions are found on the side plots.

![Alt text](docs/posterior_example.png?raw=true "Optional Title")
The posterior distribution of the Gardner toggle switch produced by StabilityFinder. The
marginal distributions of each parameter are found on the diagonal and pairwise joint distributions along
the sides.
