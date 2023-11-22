fn=$1
benchmark=$2
docker run -v $(pwd):/app/ -it evaluation-harness-multiple python3 main.py \
    --model bigcode/santacoder \
    --tasks multiple-${benchmark}-java \
    --load_generations_path /app/results/$fn/$benchmark/generations.json \
    --metric_output_path /app/results/$fn/$benchmark/eval_v3.json \
    --allow_code_execution  \
    --temperature 0.2 \
    --n_samples 200
