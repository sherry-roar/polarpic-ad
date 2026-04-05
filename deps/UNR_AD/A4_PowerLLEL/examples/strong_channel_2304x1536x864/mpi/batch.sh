#!/bin/bash

# $1: test path
# $2: executable
# $3: array of case names
# $4: test rounds
# $5: nnodes (-N)
# $6: ntasks (-n)
# $7: cpu per tasks (-c)
run_test_case(){
cases_arr=$3

for icase in ${cases_arr[*]}
do
  cd $1/$icase

  for ((i=1; i<=$4; i++))
  do
    echo ">>> Start  test "$2"-"$icase" Round "$i": -N "$5" -n "$6" -c "$7
    OMP_NUM_THREADS=$7 srun -N $5 -n $6 -c $7 ./$2 2>&1 | tee run.$i.log
    if [ -f "timing.summary" ];then
      mv timing.summary timing.summary.$i
    fi
    echo ">>> Finish test "$2"-"$icase" Round "$i"!"
  done

done
}

base=$(pwd)
exe=../../bin/PowerLLEL_cpu.mpi
nrounds=3

# 192 Nodes, 4608 Cores
cases=(384x12)
run_test_case $base $exe "${cases[*]}" $nrounds 192 4608 1

# 96 Nodes, 2304 Cores
cases=(192x12 96x24 48x48)
run_test_case $base $exe "${cases[*]}" $nrounds 96 2304 1

nrounds=2

# 48 Nodes, 1152 Cores
cases=(96x12 48x24)
run_test_case $base $exe "${cases[*]}" $nrounds 48 1152 1

# 24 Nodes, 576 Cores
cases=(48x12 24x24)
run_test_case $base $exe "${cases[*]}" $nrounds 24 576 1

# 12 Nodes, 288 Cores
cases=(24x12)
run_test_case $base $exe "${cases[*]}" $nrounds 12 288 1
