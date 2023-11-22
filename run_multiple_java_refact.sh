# checkpoints_batch_size_512/checkpoint-3000
output_dir=$1
benchmark=$2
mkdir $output_dir 
accelerate launch  main.py \
    --model smallcloudai/Refact-1_6B-fim  \
    --tasks multiple-${benchmark}-java  \
    --max_length_generation 650 \
    --temperature 0.2   \
    --do_sample True  \
    --n_samples 100  \
    --batch_size 32  \
    --trust_remote_code \
    --generation_only \
    --save_generations \
    --save_generations_path results/$output_dir/${benchmark}/generations.json
