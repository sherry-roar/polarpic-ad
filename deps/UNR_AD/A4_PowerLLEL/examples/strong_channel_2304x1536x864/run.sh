#!/bin/bash

# Description: This script evaluate the STRONG scalability of PowerLLEL on TH2A
# Details: All the input and binary will be copied to a separated directory, firstly.
#          Thus you can modify your code while it is pending or running.

set -e

cd $(dirname "$BASH_SOURCE")
export CALL_BY_RUN_SH=1

INPUT_DIR=$(pwd) 
export RUN_DIR="${INPUT_DIR}/../../runspace/`date +%Y-%m-%d_%H-%M-%S`_`hostname`_strong"
cp -r $INPUT_DIR $RUN_DIR
cp -r ${INPUT_DIR}/../../build/install/bin $RUN_DIR/
cd $RUN_DIR/

export TARGET_JOBID=69553
./batch.sh
