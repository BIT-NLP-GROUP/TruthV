DATASETS='tqa csqa2 qasc swag hellaswag siqa piqa cosmosqa cicero cicero2'
# DATASETS='tqa'
SAVE_PATH=results/likelihood

# MODEL_PATH=src/models/google/gemma-2-2b-it
# MODEL_PATH=src/models/meta-llama/Llama-2-7b-chat-hf
# MODEL_PATH=src/models/meta-llama/Llama-3.2-3B-Instruct
MODEL_PATH=src/models/Qwen/Qwen3-4B


for DATA in $DATASETS
do
    python TruthV/baseline_likelihood.py \
        --model_path $MODEL_PATH \
        --save_path $SAVE_PATH \
        --dataset $DATA
done

