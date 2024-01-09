#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(dirname `pwd`/`find . | grep libcudnn_ops_infer.so.8`):$(dirname `pwd`/`find . | grep libcublas.so.11`)

export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

source venv/bin/activate
