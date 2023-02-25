run_server:
	sbatch --gpus=1 run.sh 660 24 4 0.15

run_pc:
	python train.py -e=660 -b=24 -n=4 -lr=0.15