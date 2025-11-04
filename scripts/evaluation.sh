torchrun --standalone --nproc_per_node=2 --master_port=22222 \
    sample.py \
    --predictor_path=0 \
    --batch=128 \
    --seeds="0-49999"

# FID evaluation
python fid.py calc --images=path/to/images --ref=path/to/fid/stat