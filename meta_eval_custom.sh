#!/bin/bash
gen_path=java_sft_jaeseong
benchmark=mbpp


cnt=0
until [ $cnt -ge 10 ]
do
#     echo $gen_path multiple_java-testcase_${cnt}.pkl
    ./eval_custom_multiple_java_sft.sh $gen_path multiple_java-testcase_${cnt}.pkl $benchmark eval_custom_${cnt}.json
    ./run_rm_container.sh
    cnt=$(expr $cnt + 1)
done
