#!/bin/bash
# +
benchmark=$1
./run_multiple_java_lora.sh checkpoints_batch_size_512/checkpoint-6000  java_lora_lr_5e-5_6000 $benchmark

peft_path=checkpoints_batch_size_512_lr_1e-5/checkpoint-
output_dir=java_lora_lr_1e-5_
tail=00
# -

cnt=0
until [ $cnt -ge 30 ]
do
    cnt=$(expr $cnt + 1)
    #echo $peft_path
    #echo $peft_path$cnt$tail
    ./run_multiple_java_lora.sh $peft_path$cnt$tail $output_dir$cnt$tail $benchmark
    #./eval_humaneval_java.sh $(output_dir)$(cnt)00
done
