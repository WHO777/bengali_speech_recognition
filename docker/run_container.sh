#!/bin/bash
docker run \
-p 8869:8869 \
--gpus all \
--runtime=nvidia \
-v $DATASETS_ROOT:/app/datasets \
-v $KAGGLE_ROOT/bengali_speech_recognition:/app \
--network="host" \
--shm-size=5g \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
--name bengali_speech_recognition \
-it nvcr.io/nvidia/tensorflow:23.01-tf2-py3 /bin/bash
