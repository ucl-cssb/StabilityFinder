export PYTHONPATH=$PYTHONPATH:/home/ucbtle1/cuda-sim-code


exe=/home/ucbtle1/work/StabilityChecker/dev/StabilityChecker_stable/export_ICs_matrix/StabilityFinder/stabilityfinder


python $exe/my_abc.py -i input_file.xml -o results -l log.log;
