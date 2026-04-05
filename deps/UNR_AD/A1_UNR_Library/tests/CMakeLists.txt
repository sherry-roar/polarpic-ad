set(LAUNCH_2_CMD
    /usr/bin/mpirun -n 2)
message(STATUS "LAUNCH_1_CMD is empty")
message(STATUS "LAUNCH_2_CMD =" mpirun -n 2)

add_subdirectory(channel)
add_subdirectory(uri)
add_subdirectory(unr)