!=======================================================================
! This is part of the 2DECOMP&FFT library
! 
! 2DECOMP&FFT is a software framework for general-purpose 2D (pencil) 
! decomposition. It also implements a highly scalable distributed
! three-dimensional Fast Fourier Transform (FFT).
!
! Copyright (C) 2009-2012 Ning Li, the Numerical Algorithms Group (NAG)
!
!=======================================================================

! This file contains the routines that transpose data from Z to Y pencil

#ifdef _CUDA
  subroutine transpose_z_to_y_real_d(src, dst, opt_decomp)

    implicit none

    real(mytype), dimension(:,:,:), intent(IN) :: src
    real(mytype), dimension(:,:,:), intent(OUT) :: dst
    TYPE(DECOMP_INFO), intent(IN), optional :: opt_decomp
    attributes( managed ) :: src, dst
    TYPE(DECOMP_INFO) :: decomp

    integer :: s1,s2,s3,d1,d2,d3
    integer :: ierror, istat, m, i1, i2, pos
    integer :: iter, dest, sorc, pow2
#ifdef NVTX
    call nvtxStartRange("tranZY",5)
#endif
    if (present(opt_decomp)) then
       decomp = opt_decomp
    else
       decomp = decomp_main
    end if

    s1 = SIZE(src,1)
    s2 = SIZE(src,2)
    s3 = SIZE(src,3)
    d1 = SIZE(dst,1)
    d2 = SIZE(dst,2)
    d3 = SIZE(dst,3)

#ifdef PIPELINE_TRANS
    if(IAND(dims(2),dims(2)-1)==0) then
      pow2 = 1
    else
      pow2 = 0
    endif

    ! rearrange source array as send buffer
    do iter=1,dims(2)-1
       if( pow2 ) then
         dest = IEOR(row_rank,iter)
       else
         dest = mod(row_rank + iter, dims(2))
       endif
       m = dest
       pos = decomp%z2disp(m) + 1
       istat = cudaMemcpyAsync( work1_r_d(pos), src(1,1,decomp%z2idx(m)), decomp%z2cnts(m), a2a_d2h )
       istat = cudaEventRecord( a2a_event(iter), a2a_d2h )
    end do

    ! self
    m = row_rank
    istat = cudaMemcpy2DAsync( dst(1,decomp%y2idx(m),1), d1*d2, src(1,1,decomp%z2idx(m)), d1*(decomp%y2dist(m)), d1*(decomp%y2dist(m)), d3, stream=a2a_comp )

    do iter=1,dims(2)-1
      if( pow2 ) then
        sorc = IEOR(row_rank,iter)
      else
        sorc = mod(row_rank - iter + dims(2), dims(2))
      endif
      m = sorc
      call MPI_IRECV( work2_r_d(decomp%y2disp(m)+1), decomp%y2cnts(m), real_type, m, 0, DECOMP_2D_COMM_ROW, a2a_requests(iter), ierror)
    end do

    do iter=1,dims(2)-1
       if( pow2 ) then
          dest = IEOR(row_rank,iter)
          sorc = dest
       else
          dest = mod(row_rank + iter, dims(2))
          sorc = mod(row_rank - iter + dims(2), dims(2))
       endif
       m = dest
       istat = cudaEventSynchronize( a2a_event(iter) )
#ifdef NVTX
       call nvtxStartRangeAsync("MPI",iter)
#endif
       call MPI_SEND( work1_r_d(decomp%z2disp(m)+1), decomp%z2cnts(m), real_type, m, 0, DECOMP_2D_COMM_ROW, ierror)
#ifdef NVTX
       call nvtxEndRangeAsync
#endif
       call MPI_WAIT(a2a_requests(iter), MPI_STATUS_IGNORE, ierror)
       m = sorc
       pos = decomp%y2disp(m) + 1
       istat = cudaMemcpy2DAsync( dst(1,decomp%y2idx(m),1), d1*d2, work2_r_d(pos), d1*(decomp%y2dist(m)), d1*(decomp%y2dist(m)), d3, stream=a2a_comp )
    end do
    istat = cudaEventRecord( a2a_event(0), 0 )
    istat = cudaEventSynchronize( a2a_event(0) )
#else


    call RGPTLSTART('----mem_split_zy')

    ! istat = cudaMemcpy( work1_r, src, s1*s2*s3, cudaMemcpyDeviceToHost )
    istat = cudaMemcpy( work1_r_d, src, s1*s2*s3 )

    call RGPTLSTOP('----mem_split_zy')
    call RGPTLSTART('----mpi_alltoall_zy')

    call MPI_ALLTOALLV(work1_r_d, decomp%z2cnts, decomp%z2disp, &
         real_type, work2_r_d, decomp%y2cnts, decomp%y2disp, &
         real_type, DECOMP_2D_COMM_ROW, ierror)

    call RGPTLSTOP('----mpi_alltoall_zy')
    call RGPTLSTART('----mem_merge_zy')

    ! rearrange receive buffer
    do m=0,dims(2)-1
       if (m==0) then
          i1 = 1
          i2 = decomp%y2dist(0)
       else
          i1 = i2+1
          i2 = i1+decomp%y2dist(m)-1
       end if
       pos = decomp%y2disp(m) + 1
    !    istat = cudaMemcpy2D( dst(1,i1,1), d1*d2, work2_r(pos), d1*(i2-i1+1), d1*(i2-i1+1), d3, cudaMemcpyHostToDevice )
       istat = cudaMemcpy2D( dst(1,i1,1), d1*d2, work2_r_d(pos), d1*(i2-i1+1), d1*(i2-i1+1), d3 )
    end do

    call RGPTLSTOP('----mem_merge_zy')

#endif

#ifdef NVTX
    call nvtxEndRange
#endif

    return
  end subroutine transpose_z_to_y_real_d
#endif

  subroutine transpose_z_to_y_real(src, dst, opt_decomp)

    implicit none
    
    real(mytype), dimension(:,:,:), intent(IN) :: src
    real(mytype), dimension(:,:,:), intent(OUT) :: dst
    TYPE(DECOMP_INFO), intent(IN), optional :: opt_decomp

    TYPE(DECOMP_INFO) :: decomp
    
    integer :: s1,s2,s3,d1,d2,d3
    integer :: ierror

    if (present(opt_decomp)) then
       decomp = opt_decomp
    else
       decomp = decomp_main
    end if

    s1 = SIZE(src,1)
    s2 = SIZE(src,2)
    s3 = SIZE(src,3)
    d1 = SIZE(dst,1)
    d2 = SIZE(dst,2)
    d3 = SIZE(dst,3)

    call RGPTLSTART('----mem_split_zy')

    ! rearrange source array as send buffer
#ifdef EVEN
    if (.not. decomp%even) then
       call mem_split_zy_real(src, s1, s2, s3, work1_r, dims(2), &
            decomp%z2dist, decomp)
    end if
#else
    ! note the src array is suitable to be a send buffer
    ! so no split operation needed
#endif

    call RGPTLSTOP('----mem_split_zy')
    call RGPTLSTART('----mpi_alltoall_zy')
    
#ifdef EVEN
    if (decomp%even) then
       call MPI_ALLTOALL(src, decomp%z2count, &
            real_type, work2_r, decomp%y2count, &
            real_type, DECOMP_2D_COMM_ROW, ierror)
    else
       call MPI_ALLTOALL(work1_r, decomp%z2count, &
            real_type, work2_r, decomp%y2count, &
            real_type, DECOMP_2D_COMM_ROW, ierror)
    end if
#else
    call MPI_ALLTOALLV(src, decomp%z2cnts, decomp%z2disp, &
         real_type, work2_r, decomp%y2cnts, decomp%y2disp, &
         real_type, DECOMP_2D_COMM_ROW, ierror)
#endif

    call RGPTLSTOP('----mpi_alltoall_zy')
    call RGPTLSTART('----mem_merge_zy')

    ! rearrange receive buffer
    call mem_merge_zy_real(work2_r, d1, d2, d3, dst, dims(2), &
         decomp%y2dist, decomp)

    call RGPTLSTOP('----mem_merge_zy')
    
    return
  end subroutine transpose_z_to_y_real

#if 0
  subroutine transpose_z_to_y_complex(src, dst, opt_decomp)

    implicit none
    
    complex(mytype), dimension(:,:,:), intent(IN) :: src
    complex(mytype), dimension(:,:,:), intent(OUT) :: dst
    TYPE(DECOMP_INFO), intent(IN), optional :: opt_decomp

    TYPE(DECOMP_INFO) :: decomp
    
    integer :: s1,s2,s3,d1,d2,d3
    integer :: ierror

    if (present(opt_decomp)) then
       decomp = opt_decomp
    else
       decomp = decomp_main
    end if

    s1 = SIZE(src,1)
    s2 = SIZE(src,2)
    s3 = SIZE(src,3)
    d1 = SIZE(dst,1)
    d2 = SIZE(dst,2)
    d3 = SIZE(dst,3)
    
    ! rearrange source array as send buffer
#ifdef EVEN
    if (.not. decomp%even) then
       call mem_split_zy_complex(src, s1, s2, s3, work1_c, dims(2), &
            decomp%z2dist, decomp)
    end if
#else
    ! note the src array is suitable to be a send buffer
    ! so no split operation needed
#endif
    
#ifdef EVEN
    if (decomp%even) then
       call MPI_ALLTOALL(src, decomp%z2count, &
            complex_type, work2_c, decomp%y2count, &
            complex_type, DECOMP_2D_COMM_ROW, ierror)
    else
       call MPI_ALLTOALL(work1_c, decomp%z2count, &
            complex_type, work2_c, decomp%y2count, &
            complex_type, DECOMP_2D_COMM_ROW, ierror)
    end if
#else
    call MPI_ALLTOALLV(src, decomp%z2cnts, decomp%z2disp, &
         complex_type, work2_c, decomp%y2cnts, decomp%y2disp, &
         complex_type, DECOMP_2D_COMM_ROW, ierror)
#endif

    ! rearrange receive buffer
    call mem_merge_zy_complex(work2_c, d1, d2, d3, dst, dims(2), &
         decomp%y2dist, decomp)

    return
  end subroutine transpose_z_to_y_complex
#endif

  ! pack/unpack ALLTOALL(V) buffers

  subroutine mem_split_zy_real(in,n1,n2,n3,out,iproc,dist,decomp)

    implicit none

    integer, intent(IN) :: n1,n2,n3
    real(mytype), dimension(n1,n2,n3), intent(IN) :: in
    real(mytype), dimension(*), intent(OUT) :: out
    integer, intent(IN) :: iproc
    integer, dimension(0:iproc-1), intent(IN) :: dist
    TYPE(DECOMP_INFO), intent(IN) :: decomp
    
    integer :: i,j,k, m,i1,i2,pos
    integer :: k1,k2
    integer :: offset = 0
    !$ integer :: tid, nt, ksz

    !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(m,k,j,i,i1,i2,pos,tid,nt,k1,k2,ksz,offset)
    !$ tid = OMP_GET_THREAD_NUM()
    !$ nt  = OMP_GET_NUM_THREADS()
    do m=0,iproc-1
       if (m==0) then 
          i1 = 1
          i2 = dist(0)
       else
          i1 = i2+1
          i2 = i1+dist(m)-1
       end if

       k1 = i1
       k2 = i2
       !$ call distribute_among_threads('f', nt, tid, dist(m), 1, ksz, k1, k2)
       !$ k1 = i1 + k1 - 1
       !$ k2 = i1 + k2 - 1

       !$ offset = n1 * n2 * (k1-i1)
#ifdef EVEN
       pos = m * decomp%z2count + 1 + offset
#else
       pos = decomp%z2disp(m) + 1 + offset
#endif

       do k=k1,k2
          do j=1,n2
             do i=1,n1
                out(pos) = in(i,j,k)
                pos = pos + 1
             end do
          end do
       end do
    end do
    !$OMP END PARALLEL

    return
  end subroutine mem_split_zy_real

#if 0
  subroutine mem_split_zy_complex(in,n1,n2,n3,out,iproc,dist,decomp)

    implicit none

    integer, intent(IN) :: n1,n2,n3
    complex(mytype), dimension(n1,n2,n3), intent(IN) :: in
    complex(mytype), dimension(*), intent(OUT) :: out
    integer, intent(IN) :: iproc
    integer, dimension(0:iproc-1), intent(IN) :: dist
    TYPE(DECOMP_INFO), intent(IN) :: decomp
    
    integer :: i,j,k, m,i1,i2,pos

    do m=0,iproc-1
       if (m==0) then 
          i1 = 1
          i2 = dist(0)
       else
          i1 = i2+1
          i2 = i1+dist(m)-1
       end if

#ifdef EVEN
       pos = m * decomp%z2count + 1
#else
       pos = decomp%z2disp(m) + 1
#endif

       do k=i1,i2
          do j=1,n2
             do i=1,n1
                out(pos) = in(i,j,k)
                pos = pos + 1
             end do
          end do
       end do
    end do

    return
  end subroutine mem_split_zy_complex
#endif

  subroutine mem_merge_zy_real(in,n1,n2,n3,out,iproc,dist,decomp)

    implicit none
    
    integer, intent(IN) :: n1,n2,n3
    real(mytype), dimension(*), intent(IN) :: in
    real(mytype), dimension(n1,n2,n3), intent(OUT) :: out
    integer, intent(IN) :: iproc
    integer, dimension(0:iproc-1), intent(IN) :: dist
    TYPE(DECOMP_INFO), intent(IN) :: decomp
    
    integer :: i,j,k, m,i1,i2, pos
    integer :: k1,k2
    integer :: offset = 0
    !$ integer :: tid, nt, ksz

    k1 = 1
    k2 = n3
    
    !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(m,k,j,i,i1,i2,pos,tid,nt,k1,k2,ksz,offset)
    !$ tid = OMP_GET_THREAD_NUM()
    !$ nt  = OMP_GET_NUM_THREADS()
    !$ call distribute_among_threads('f', nt, tid, n3, 1, ksz, k1, k2)
    do m=0,iproc-1
       if (m==0) then
          i1 = 1
          i2 = dist(0)
       else
          i1 = i2+1
          i2 = i1+dist(m)-1
       end if

       !$ offset = (i2-i1+1) * n1 * (k1-1)
#ifdef EVEN
       pos = m * decomp%y2count + 1 + offset
#else
       pos = decomp%y2disp(m) + 1 + offset
#endif

       do k=k1,k2
          do j=i1,i2
             do i=1,n1
                out(i,j,k) = in(pos)
                pos = pos + 1
             end do
          end do
       end do
    end do
    !$OMP END PARALLEL

    return
  end subroutine mem_merge_zy_real

#if 0
  subroutine mem_merge_zy_complex(in,n1,n2,n3,out,iproc,dist,decomp)
    
    implicit none
    
    integer, intent(IN) :: n1,n2,n3
    complex(mytype), dimension(*), intent(IN) :: in
    complex(mytype), dimension(n1,n2,n3), intent(OUT) :: out
    integer, intent(IN) :: iproc
    integer, dimension(0:iproc-1), intent(IN) :: dist
    TYPE(DECOMP_INFO), intent(IN) :: decomp
    
    integer :: i,j,k, m,i1,i2, pos

    do m=0,iproc-1
       if (m==0) then
          i1 = 1
          i2 = dist(0)
       else
          i1 = i2+1
          i2 = i1+dist(m)-1
       end if

#ifdef EVEN
       pos = m * decomp%y2count + 1
#else
       pos = decomp%y2disp(m) + 1
#endif

       do k=1,n3
          do j=i1,i2
             do i=1,n1
                out(i,j,k) = in(pos)
                pos = pos + 1
             end do
          end do
       end do
    end do

    return
  end subroutine mem_merge_zy_complex
#endif
