run_server:
	sbatch --gpus=8 run.sh 660 24 8 0.15

run_gpuc128:
	sbatch --gpus=8 -p gpu_c128 run.sh 660 24 8 0.15

run_pc:
	python train.py -e=660 -b=24 -n=8 -lr=0.15 --aspect_ratio_group_factor=0