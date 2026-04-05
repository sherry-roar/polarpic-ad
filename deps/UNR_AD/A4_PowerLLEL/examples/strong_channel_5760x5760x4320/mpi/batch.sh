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

# 4320 Nodes, 103680 Cores
cases=(1440x72)
run_test_case $base $exe "${cases[*]}" $nrounds 4320 103680 1

# 3456 Nodes, 82944 Cores
cases=(1152x72)
run_test_case $base $exe "${cases[*]}" $nrounds 3456 82944 1

# 1728 Nodes, 41472 Cores
cases=(576x72)
run_test_case $base $exe "${cases[*]}" $nrounds 1728 41472 1

# 864 Nodes, 20736 Cores
cases=(288x72)
run_test_case $base $exe "${cases[*]}" $nrounds 864 20736 1

nrounds=3

# 576 Nodes, 13824 Cores
cases=(192x72)
run_test_case $base $exe "${cases[*]}" $nrounds 576 13824 1

# 432 Nodes, 10368 Cores
cases=(144x72)
run_test_case $base $exe "${cases[*]}" $nrounds 432 10368 1

# 288 Nodes, 6912 Cores
cases=(96x72)
run_test_case $base $exe "${cases[*]}" $nrounds 288 6912 1
