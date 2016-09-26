

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
