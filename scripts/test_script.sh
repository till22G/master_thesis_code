#!/usr/bin/env bash

set -x
set -e

cd $( dirname "$0" ) && cd .. && cd rebuild_SimKGC
pwd
python3 main.py