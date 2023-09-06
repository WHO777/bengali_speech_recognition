#!/bin/bash
docker run \
-p 8869:8869 \
--gpus all \
--runtime=nvidia \
--mount type=bind,source=$KAGGLE_ROOT/bengali_speech_recognition,target=/app \
--network="host" \
--shm-size=5g \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
--name bengali_speech_recognition \
-it nvcr.io/nvidia/tensorflow:23.01-tf2-py3 /bin/bash
