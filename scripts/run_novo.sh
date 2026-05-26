# DATASETS='tqa csqa2 qasc swag hellaswag siqa piqa cosmosqa cicero cicero2'
DATASETS='tqa'
SAVE_PATH=results/novo

MODEL_PATH=src/models/meta-llama/Llama-2-7b-chat-hf
# MODEL_PATH=src/models/google/gemma-2-2b-it

for DATA in $DATASETS
do
    python TruthV/baseline_novo.py \
        --model_path $MODEL_PATH \
        --save_path $SAVE_PATH \
        --dataset $DATA
done