

SOLVER_FLAGS="--sampler_stu=adasde --sampler_tea=dpm --num_steps=3 --M=3 --afs=True --scale_dir=0.05 --scale_time=0.2 --gamma=0.02 --seed=0 --lr=0.2 --coslr"
SCHEDULE_FLAGS="--schedule_type=time_uniform --schedule_rho=1"
torchrun --standalone --nproc_per_node=4 --master_port=11111 \
train.py --dataset_name="cifar10" --batch=32 --total_kimg=10 $SOLVER_FLAGS $SCHEDULE_FLAGS

SOLVER_FLAGS="--sampler_stu=adasde --sampler_tea=dpm --num_steps=4 --M=3 --afs=True --scale_dir=0.05 --scale_time=0.2 --gamma=0.02 --seed=0 --lr=0.2 --coslr"
SCHEDULE_FLAGS="--schedule_type=time_uniform --schedule_rho=1"
torchrun --standalone --nproc_per_node=2 --master_port=11111 \
train.py --dataset_name="cifar10" --batch=64 --total_kimg=10 $SOLVER_FLAGS $SCHEDULE_FLAGS


SOLVER_FLAGS="--sampler_stu=adasde --sampler_tea=dpm --num_steps=3 --M=4 --afs=True --scale_dir=0.05 --scale_time=0.2 --gamma=0.02 --seed=0 --lr=0.1 --coslr --alpha=50"
SCHEDULE_FLAGS="--schedule_type=time_uniform --schedule_rho=1"
torchrun --standalone --nproc_per_node=4 --master_port=11111 \
train.py --dataset_name="ffhq" --batch=32 --total_kimg=10 $SOLVER_FLAGS $SCHEDULE_FLAGS

SOLVER_FLAGS="--sampler_stu=adasde --sampler_tea=dpm --num_steps=4 --M=3 --afs=True --scale_dir=0.05 --scale_time=0.2 --gamma=0.02 --seed=0 --lr=0.1 --coslr --alpha=50"
SCHEDULE_FLAGS="--schedule_type=time_uniform --schedule_rho=1"
torchrun --standalone --nproc_per_node=2 --master_port=11111 \
train.py --dataset_name="ffhq" --batch=32 --total_kimg=10 $SOLVER_FLAGS $SCHEDULE_FLAGS

SOLVER_FLAGS="--sampler_stu=adasde --sampler_tea=dpm --num_steps=6 --M=3 --afs=True --scale_dir=0.05 --scale_time=0.2 --gamma=0 --seed=0 --lr=0.1 --coslr --alpha=50"
SCHEDULE_FLAGS="--schedule_type=time_uniform --schedule_rho=1"
torchrun --standalone --nproc_per_node=2 --master_port=11111 \
train.py --dataset_name="ffhq" --batch=32 --total_kimg=10 $SOLVER_FLAGS $SCHEDULE_FLAGS

SOLVER_FLAGS="--sampler_stu=adasde --sampler_tea=dpm --num_steps=6 --M=3 --afs=True --scale_dir=0.05 --scale_time=0.2 --gamma=0.02 --seed=0 --lr=0.2 --coslr --alpha=50"
SCHEDULE_FLAGS="--schedule_type=time_uniform --schedule_rho=1"
torchrun --standalone --nproc_per_node=2 --master_port=11111 \
train.py --dataset_name="imagenet64" --batch=32 --total_kimg=10 $SOLVER_FLAGS $SCHEDULE_FLAGS


torchrun --standalone --nproc_per_node=2 --master_port=22222 \
    sample.py \
    --predictor_path=6 \
    --batch=128 \
    --seeds="0-49999"

python fid.py calc \
    --images="/zhaotong/EPD/samples/cifar10/adasde_nfe9_npoints_1" \
    --ref="/zhaotong/CS2/ref/cifar10-32x32.npz"