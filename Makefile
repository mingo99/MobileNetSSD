run_server:
	sbatch --gpus=${GPUS} run.sh ${GPUS} 660 24 8 0.15 ""

run_gpuc128:
	sbatch --gpus=8 -p gpu_c128 run.sh 8 660 24 8 0.15 ""

run_resume:
	sbatch --gpus=8 -p gpu_c128 run.sh 8 660 24 8 0.15 "./checkpoint/normal/checkpoint.pth"

run_mm:
	sbatch -N 2 --gres=gpu:8 --qos=gpugpu run.sh 220 24 16 0.15 ""

run_pc:
	python train.py -e=330 -b=24 -n=4 -lr=0.05 --aspect_ratio_group_factor=0 -y=2023 --eval_freq=5

run_test:
	python train.py -y=2017 --eval_freq=5 --test-only True --resume "./checkpoint/normal/best.pth"