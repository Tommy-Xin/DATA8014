#conda create --name 1dtokenizer python=3.10


conda activate 1dtokenizer
cd /root/users/jusjus/1d-tokenizer-modified

## Datasets

proxychains python /root/users/jusjus/1d-tokenizer-modified/data/convert_imagenet_to_wds.py --output_dir /mnt/znzz/jus/datasets/imagenet_sharded


## Training
 WANDB_MODE=offline accelerate launch --num_machines=1 --num_processes=4 --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network scripts/train_titok.py config=configs/training/stage1/titok_b128.yaml \
    experiment.project="titok_b_128_stage1" \
    experiment.name="titok_b_128_stage1_run1" \
    experiment.output_dir="titok_b_128_stage1_run1" \
    training.per_gpu_batch_size=32

WANDB_MODE=offline proxychains accelerate launch --num_machines=1 --num_processes=4 --machine_rank=0 --same_network scripts/train_titok.py config=configs/training/stage1/titok_b128.yaml \
    experiment.project="titok_b_128_stage1" \
    experiment.name="titok_b_128_stage1_run1" \
    experiment.output_dir="titok_b_128_stage1_run1" \
    training.per_gpu_batch_size=32

accelerate launch --config_file /root/users/jusjus/1d-tokenizer-modified/configs/accelerate_config.json 

proxychains python scripts/train_titok.py config=configs/training/stage1/titok_b128.yaml \
    experiment.project="titok_b_128_stage1" \
    experiment.name="titok_b_128_stage1_run1" \
    experiment.output_dir="titok_b_128_stage1_run1" \
    training.per_gpu_batch_size=32

proxychains pip install --index-url https://pypi.org/simple iopath