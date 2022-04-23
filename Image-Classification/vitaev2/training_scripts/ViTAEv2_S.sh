python -m torch.distributed.launch --nnodes 2 --node_rank {NODE_ID} --master_addr {} --master_port 25901 --nproc_per_node 8 \
    ./main.py {dataset-path} --model ViTAEv2_S -b 64 --lr 5e-4 --weight-decay .05 --img-size 224 --workers 8

python -m torch.distributed.launch --master_port 25901 --nproc_per_node 8 \
    ./main.py {dataset-path} --model ViTAEv2_S -b 128 --lr 5e-4 --weight-decay .05 --img-size 224 --workers 8
