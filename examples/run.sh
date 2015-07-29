#export PYTHONPATH=$PYTHONPATH:/home/ucbtle1/.local/lib/python2.7/site-packages/stabilitychecker
<<<<<<< Updated upstream
#exe=/home/ucbtle1/.local/lib/python2.7/site-packages/stabilitychecker

exe=/home/ucbtle1/work/StabilityChecker/dev/StabilityChecker_stable/StabilityChecker/stabilitychecker

=======
exe=/home/ucbtle1/work/StabilityChecker/dev/StabilityChecker_stable/StabilityChecker/stabilitychecker
>>>>>>> Stashed changes
export CUDA_DEVICE=6

python $exe/read_input.py; python $exe/my_abc.py; #Rscript $exe/plot_posterior.R;
