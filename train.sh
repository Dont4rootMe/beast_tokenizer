PYTHONPATH="${PYTHONPATH}:$(pwd)" python train/train_beast.py \
    --batch-size 32 \
    --num-basis 50 \
    --vocab-size 1000 \
    --degree 0 \
    --device cpu \
    --fit-beast-max-samples 5000 \
    --fit-bpe-max-samples 25000 \
    --bpe-vocab-size 2048 \
    --beast-checkpoint-dir beast_tokenizer_checkpoint \
    --bpe-checkpoint-dir beast_bpe_tokenizer_checkpoint \
    --eval-results-dir eval_results \
    --max-eval-samples 2500

