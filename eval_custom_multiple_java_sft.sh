fn=$1
tc=$2
benchmark=$3
out=$4
docker run -v $(pwd):/app/ -it evaluation-harness-multiple python3 main.py \
    --model bigcode/santacoder \
    --tasks multiple-${benchmark}-java \
    --multiple_benchmark ${benchmark} \
    --load_generations_path /app/results/$fn/$benchmark/generations.json \
    --metric_output_path /app/results/$fn/$benchmark/$out \
    --allow_code_execution  \
    --temperature 0.2 \
    --n_samples 200 \
    --custom_test /app/custom_testcase/$tc
