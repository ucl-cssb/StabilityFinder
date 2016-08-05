module load cuda/4.2.9
module load pycuda/2012.1

module load R
##module load cuda-sim

#BSUB -o log.std_tri
#BSUB -e err.std_tri
#BSUB -W 100:00

export PYTHONPATH=$PYTHONPATH:/home/ucl/eisuc058/code/abc-sysbio
export PYTHONPATH=$PYTHONPATH:/home/ucl/eisuc058/code/cuda-sim

exe=/home/ucl/eisuc058/work/StabilityChecker/stabilitychecker
export PYTHONPATH=$PYTHONPATH:/home/ucbtle1/cuda-sim-code

python $exe/my_abc.py -i input_file_std_tri.xml -o results_tri -l std_tri.log;# Rscript plot_posterior.R;
