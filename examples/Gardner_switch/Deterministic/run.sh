export PYTHONPATH=$PYTHONPATH:/home/ucbtle1/cuda-sim-code
#exe=/home/ucbtle1/.local/lib/python2.7/site-packages/stabilitychecker

exe=/home/ucbtle1/work/StabilityChecker/dev/StabilityChecker_stable/export_ICs_matrix/StabilityFinder/stabilityfinder


python $exe/my_abc.py -i input_file.xml -o results_deter_high_mean_high_var -l gardner_deter_high_mean_high_var.log;# Rscript plot_posterior.R;
