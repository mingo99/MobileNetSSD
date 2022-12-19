run_server:
	sbatch --gpus=1 run.sh 100 128 0.01

run_pc:
	python train.py -e=$1 -b=$2 -lr=$3