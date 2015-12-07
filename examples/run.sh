export PYTHONPATH=$PYTHONPATH:/home/ucbtle1/cuda-sim-code
#exe=/home/ucbtle1/.local/lib/python2.7/site-packages/stabilitychecker

exe=/home/ucbtle1/work/StabilityChecker/dev/StabilityChecker_stable/StabilityChecker/stabilitychecker

python $exe/my_abc.py -i input_file.xml -o results_test2 -l gard_test2.log;# Rscript plot_posterior.R;
