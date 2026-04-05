module mod_type

    implicit none

    ! Single precision or double precision (default)
#ifdef _SINGLE_PREC
    integer, parameter :: fp = selected_real_kind(6)
#else
    integer, parameter :: fp = selected_real_kind(15)
#endif

#ifdef SP_POIS
    integer, parameter :: fp_pois = selected_real_kind(6)
#else
    integer, parameter :: fp_pois = fp
#endif

    integer, parameter :: i8 = selected_int_kind(18)

    ! The derived type for recording the info. of statistics processing
    type stat_info_t
        integer :: nts, nte
        integer :: nspl
    end type

end module mod_type