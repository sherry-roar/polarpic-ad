if (USE_GLEX)
    link_libraries(unr)

    add_executable(01_reg_same_loc 01_reg_same_loc.c)
    install(TARGETS 01_reg_same_loc)
    add_test(NAME 01_reg_same_loc COMMAND 01_reg_same_loc)
endif()