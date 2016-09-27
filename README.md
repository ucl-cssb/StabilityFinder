

Installation
-------------

StabilityFinder has been developed to work on a Linux operating system, with GPU support.
The following dependencies are essential for the successful implementation of the package:
  
 -numpy
 -logging
 -cuda-sim
 -libsbml
 -R
 -ggplot2
 -pycuda
 
 
$ cd StabilityFinder
$ python setup.py install
 
This will copy the module StabilityChecker into the lib/pythonXX/site-packages directory,
where XX corresponds to the python version that was used. An exe=\<dir\> variable should also be added
to the run.sh script pointing the script to the right package installation. Alternatively the path to the
module can be added to the top of the run.sh file without the need for installation. The path to the
cuda sim module must be added to the top of the run.sh file.

$ export PATH=\<dir\>:$PATH


Using the package
-------------------
###The working directory

The working directory must contain the following les. Each one is described in detail in the section
following.
input file.xml The user input file
run.sh The shell script that will initialise the scripts
plot posterior.R The R code to plot the results
model file This can take two formats:
model.cu Cuda file of the model that is required to have this name
SBML model This can have any name, as long as this is provided in the input_file.xml.
Cuda-sim will create the model.cu file using this SBML model.



###The input file

The input.xml file contains all the necessary information that is not contained in the model itself. This
file must follow the specific format showed in the 'Examples' folder. No whitespaces are to be included
between the tags.

####Required arguments

* **epsilons** The acceptable distance from the desired value.
  * **epsilon t** Total variance in the data
    * **start** The initial cut-off value for the distance from the desired values.
    * **end** The final cut-off value for the distance from the desired values.
  * **epsilon vcl** Within-cluster variance
    * **start** The initial cut-off value for the distance from the desired values.
    * **end** The final cut-off value for the distance from the desired values.
  * **eps cl** Number of clusters
    * **start** The initial cut-off value for the distance from the desired values.
    * **end** The final cut-off value for the distance from the desired values.
* Desired final values The desired values the algorithm will converge to.
  * **number of clusters** The number of clusters in the data,describing its stability.
  * **total variance** The total variance in the data.
  * **cluster variance** The within-cluster variance in the data.
* **particles** The number of accepted parameter sets required to proceed to the next iteration.
* **number to sample** Number of parameter sets to sample from for iteration. This needs to be greater or equal to the particles.
* **initial conditions** samples Number of initial condition sets to sample for each parameter set at each iteration.
* **alpha** A value that describes the step size for the incremental reduction of the epsilon at each iteration.
This should be set to a small value (a = 0.1) to speed up the algorithm.
* **times** The time points for the simulation are given as a whitespace delimited list.
* **species numb to fit** Which two species numbers to be t by the algorithm. The should be whitespace
delimited.
* **stoch determ** The type of simulation to be used. The two options are stochastic or deterministic.
* **model file** Whether the model provided is in SBML or cuda format. The two options are sbml and cuda.
* **sbml name** The name of the SBML file that will be provided. Note that if a cuda file is provided, that
must be named model.cu.
* **parameters** The parameters included in the model.
  * **item** One item corresponds to one parameter. The number and order of the items must strictly correspond to those of the parameters in the model provided.
  * **name** The name of the parameter. Note that the parameters are read by order and not name from the model. The name provided here is used in the plotting module.
* **distribution** From what distribution to sample values from. The two options are constant and uniform.
  * **start** The lower limit of the distribution to sample from.
  * **end** The upper limit of the distribution to sample from. Note that if the distribution is
* **constant** this value is not read.
* **initial conditions** The species included in the model.
* **item** One item corresponds to one species. The number and order of the items must strictly correspond to those of the species in the model provided.
* **name** The name of the species. Note that the species are read by order and not name from the model.
* **distribution** From what distribution to sample values from. The two options are constant and uniform. Strictly two species must be set to uniform and the rest constant.
  * **start** The lower limit of the distribution to sample from.
  * **end** The upper limit of the distribution to sample from. Note that if the distribution is constant this value is not read.


