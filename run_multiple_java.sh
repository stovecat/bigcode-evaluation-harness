# checkpoints_batch_size_512/checkpoint-3000
output_dir=$1
benchmark=$2
mkdir 
accelerate launch  main.py \
    --model bigcode/santacoder  \
    --tasks multiple-${benchmark}-java  \
    --max_length_generation 650 \
    --temperature 0.2   \
    --do_sample True  \
    --n_samples 200  \
    --batch_size 64  \
    --trust_remote_code \
    --generation_only \
    --save_generations \
    --save_generations_path results/$output_dir/$benchmark/generations.json
