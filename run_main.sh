DEVICE='cuda:1'
METHOD=max

# DATASETS='tqa csqa2 qasc swag hellaswag siqa piqa cosmosqa cicero cicero2'
DATASETS='csqa2'
# SAVE_PATH=results/$METHOD
MODEL_PATH=src/models/google/gemma-2-2b-it
# MODEL_PATH=src/models/meta-llama/Llama-2-7b-chat-hf
# MODEL_PATH=src/models/meta-llama/Llama-3.2-3B-Instruct
# MODEL_PATH=src/models/Qwen/Qwen3-4B

for DATA in $DATASETS
do
    python TruthV/main.py \
        --model_path $MODEL_PATH \
        --dataset $DATA \
        --method $METHOD \
        --device $DEVICE

done

