#!/bin/bash
# 挂入后台运行
# 获取当前时间
export time_str=$(date "+%Y_%m_%d")
nohup gunicorn -c gunicorn.conf.py main:app > ${time_str}.log 2>&1 &

    