run_server:
	sbatch --gpus=1 run.sh 100 128 0.01

run_pc:
	python train.py -e=10 -b=16 -lr=0.0001