#!/bin/bash
benchmark=$1
peft_path=checkpoints_batch_size_512_lr_1e-5/checkpoint-
output_dir=java_lora_lr_1e-5_
tail=00


cnt=0
until [ $cnt -ge 30 ]
do
    cnt=$(expr $cnt + 1)
    ./eval_multiple_java.sh $output_dir$cnt$tail $benchmark
    ./run_rm_container.sh
done
