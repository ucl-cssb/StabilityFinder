module load cuda/4.2.9
module load pycuda/2012.1

module load R
##module load cuda-sim

#BSUB -o log.sp
#BSUB -e err.sp
#BSUB -W 100:00

export PYTHONPATH=$PYTHONPATH:/home/ucl/eisuc058/code/abc-sysbio
export PYTHONPATH=$PYTHONPATH:/home/ucl/eisuc058/code/cuda-sim

exe=/home/ucl/eisuc058/work/StabilityChecker/stabilitychecker
export PYTHONPATH=$PYTHONPATH:/home/ucbtle1/cuda-sim-code


python $exe/my_abc.py -i input_file_SP.xml -o results_SP -l SP.log;# Rscript plot_posterior.R;
