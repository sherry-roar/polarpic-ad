module mod_variables
    use mod_type,  only: fp
    use mod_utils, only: abort, setZero
#ifdef _CUDA
    use cudafor
    use mod_device, only: istat, mydev
#endif

    implicit none
    ! make everything public unless declared private
    public

    ! main arrays
    real(fp), allocatable, dimension(:,:,:), save :: u, v, w, p
    real(fp), allocatable, dimension(:,:,:), save :: u1, v1, w1
    !@cuf attributes(managed) :: u, v, w, p
    !@cuf attributes(managed) :: u1, v1, w1
    
    contains
    subroutine allocVariables(nhalo, sz)
        implicit none
        integer, dimension(6), intent(in) :: nhalo
        integer, dimension(3), intent(in) :: sz

        integer :: istatus

        allocate(u (1-nhalo(1):sz(1)+nhalo(2), 1-nhalo(3):sz(2)+nhalo(4), 1-nhalo(5):sz(3)+nhalo(6)), stat=istatus)
        if (istatus /= 0) call abort(101, "allocVariables: Out of memory when allocating major variables!")
        call setZero(u )
        allocate(v (1-nhalo(1):sz(1)+nhalo(2), 1-nhalo(3):sz(2)+nhalo(4), 1-nhalo(5):sz(3)+nhalo(6)), stat=istatus)
        if (istatus /= 0) call abort(101, "allocVariables: Out of memory when allocating major variables!")
        call setZero(v )
        allocate(w (1-nhalo(1):sz(1)+nhalo(2), 1-nhalo(3):sz(2)+nhalo(4), 1-nhalo(5):sz(3)+nhalo(6)), stat=istatus)
        if (istatus /= 0) call abort(101, "allocVariables: Out of memory when allocating major variables!")
        call setZero(w )
        allocate(u1(1-nhalo(1):sz(1)+nhalo(2), 1-nhalo(3):sz(2)+nhalo(4), 1-nhalo(5):sz(3)+nhalo(6)), stat=istatus)
        if (istatus /= 0) call abort(101, "allocVariables: Out of memory when allocating major variables!")
        call setZero(u1)
        allocate(v1(1-nhalo(1):sz(1)+nhalo(2), 1-nhalo(3):sz(2)+nhalo(4), 1-nhalo(5):sz(3)+nhalo(6)), stat=istatus)
        if (istatus /= 0) call abort(101, "allocVariables: Out of memory when allocating major variables!")
        call setZero(v1)
        allocate(w1(1-nhalo(1):sz(1)+nhalo(2), 1-nhalo(3):sz(2)+nhalo(4), 1-nhalo(5):sz(3)+nhalo(6)), stat=istatus)
        if (istatus /= 0) call abort(101, "allocVariables: Out of memory when allocating major variables!")
        call setZero(w1)
        allocate(p (0:sz(1)+1, 0:sz(2)+1, 0:sz(3)+1), stat=istatus)
        if (istatus /= 0) call abort(101, "allocVariables: Out of memory when allocating major variables!")
        call setZero(p )

        !@cuf istat = cudaMemAdvise(  u, size( u), cudaMemAdviseSetPreferredLocation, mydev )
        !@cuf istat = cudaMemAdvise(  v, size( v), cudaMemAdviseSetPreferredLocation, mydev )
        !@cuf istat = cudaMemAdvise(  w, size( w), cudaMemAdviseSetPreferredLocation, mydev )
        !@cuf istat = cudaMemAdvise( u1, size(u1), cudaMemAdviseSetPreferredLocation, mydev )
        !@cuf istat = cudaMemAdvise( v1, size(v1), cudaMemAdviseSetPreferredLocation, mydev )
        !@cuf istat = cudaMemAdvise( w1, size(w1), cudaMemAdviseSetPreferredLocation, mydev )
        !@cuf istat = cudaMemAdvise(  p, size( p), cudaMemAdviseSetPreferredLocation, mydev )
    
        return
    end subroutine allocVariables

    subroutine freeVariables()
        implicit none

        deallocate(u )
        deallocate(v )
        deallocate(w )
        deallocate(u1)
        deallocate(v1)
        deallocate(w1)
        deallocate(p )

        return
    end subroutine freeVariables

end module mod_variables