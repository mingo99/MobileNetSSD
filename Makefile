run_server:
	sbatch --gpus=${GPUS} run.sh 100 128 0.01

run_pc:
	python train.py -e=10 -b=32 -lr=0.01