fn=$1
benchmark=$2
docker run -v $(pwd):/app/ -it evaluation-harness-multiple python3 main.py \
    --model bigcode/santacoder \
    --tasks multiple-${benchmark}-java \
    --multiple_benchmark ${benchmark} \
    --load_generations_path /app/$fn/$benchmark/generations.json \
    --metric_output_path /app/$fn/eval_${benchmark}.json \
    --allow_code_execution  \
    --temperature 0.2 \
    --n_samples 200
