#!/bin/bash
name=evaluation-harness-multiple
id_list=$(docker ps -a | awk -v i="^$name.*" '{if($2~i){print$1}}')
docker rm $id_list | tr '\n' ' '