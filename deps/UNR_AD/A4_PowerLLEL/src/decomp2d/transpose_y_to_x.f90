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

! This file contains the routines that transpose data from Y to X pencil

#ifdef _CUDA
  subroutine transpose_y_to_x_real_d(src, dst, opt_decomp)

    implicit none

    real(mytype), dimension(:,:,:), intent(IN) :: src
    real(mytype), dimension(:,:,:), intent(OUT) :: dst
    TYPE(DECOMP_INFO), intent(IN), optional :: opt_decomp
    attributes( managed ) :: src, dst
    TYPE(DECOMP_INFO) :: decomp

    integer :: s1,s2,s3,d1,d2,d3
    integer :: ierror, istat, m, i1, i2, pos
    integer :: iter, dest, sorc, pow2
    integer :: sorc_glb, dest_glb
    integer :: i, j, k, js
    
#ifdef NVTX
    call nvtxStartRange("tranYX",7)
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
    if(IAND(dims(1),dims(1)-1)==0) then
      pow2 = 1
    else
      pow2 = 0
    endif

    ! rearrange source array as send buffer
    do iter=1,dims(1)-1
       if( pow2 ) then
         dest = IEOR(col_rank,iter)
       else
         dest = mod(col_rank + iter, dims(1))
       endif
       m = dest
       pos = decomp%y1disp(m) + 1
       istat = cudaMemcpy2DAsync( work1_r_d(pos), s1*(decomp%y1dist(m)), src(1,decomp%y1idx(m),1), s1*s2, s1*(decomp%y1dist(m)),s3, stream=a2a_d2h )
       istat = cudaEventRecord( a2a_event(iter), a2a_d2h )
    end do

    ! self
    m = col_rank
    pos = decomp%x1disp(m) + 1
    if( dims(1) .eq. 1 ) then
      istat = cudaMemcpy2DAsync( dst, s1*s2, src, s1*s2, s1*s2, s3, stream=a2a_comp )
    else
      !TODO: replace these two copy with a 3D copy or custom kernel for direct src => dst
      istat = cudaMemcpy2DAsync( work2_r_d(pos), s1*(decomp%y1dist(m)), src(1,decomp%y1idx(m),1), s1*s2, s1*(decomp%y1dist(m)),s3, stream=a2a_comp )
      istat = cudaMemcpy2DAsync( dst(decomp%x1idx(m),1,1), d1, work2_r_d(pos), decomp%x1dist(m), decomp%x1dist(m), d2*d3, stream=a2a_comp )
    !   i1 = decomp%y1idx(m)
    !   i2 = decomp%y1idx(m) + decomp%y1dist(m) - 1
    !   js = decomp%x1idx(m)
    !   !$cuf kernel do(3) <<<*,*,stream=a2a_comp>>>
    !   do k=1,s3
    !   do j=i1,i2
    !   do i=1,s1
    !       dst(i+js-1, j-i1+1, k) = src(i,j,k)
    !   end do
    !   end do
    !   end do
    endif

    do iter=1,dims(1)-1
      if( pow2 ) then
        sorc = IEOR(col_rank,iter)
      else
        sorc = mod(col_rank - iter + dims(1), dims(1))
      endif
      m = sorc
      call MPI_IRECV( work2_r_d(decomp%x1disp(m)+1), decomp%x1cnts(m), real_type, m, 0, DECOMP_2D_COMM_COL, reqs_a2a(iter),ierror)
    end do

    do iter=1,dims(1)-1
       if( pow2 ) then
          dest = IEOR(col_rank,iter)
          sorc = dest
       else
          dest = mod(col_rank + iter, dims(1))
          sorc = mod(col_rank - iter + dims(1), dims(1))
       endif
       m = dest
       istat = cudaEventSynchronize( a2a_event(iter) )
#ifdef NVTX
       call nvtxStartRangeAsync("MPI",iter)
#endif
       call MPI_SEND( work1_r_d(decomp%y1disp(m)+1), decomp%y1cnts(m), real_type, m, 0, DECOMP_2D_COMM_COL, ierror)
#ifdef NVTX
       call nvtxEndRangeAsync
#endif
       call MPI_WAIT(reqs_a2a(iter), MPI_STATUS_IGNORE, ierror)
       m = sorc
       pos = decomp%x1disp(m) + 1
       istat = cudaMemcpy2DAsync( dst(decomp%x1idx(m),1,1), d1, work2_r_d(pos), decomp%x1dist(m), decomp%x1dist(m), d2*d3,stream=a2a_comp )
    end do
    istat = cudaEventRecord( a2a_event(0), 0 )
    istat = cudaEventSynchronize( a2a_event(0) )
#else

    call RGPTLSTART('----mem_split_yx')

    ! rearrange source array as send buffer
    ! Self-copy
    pos = decomp%y1disp(col_rank) + 1
    i1 = decomp%y1st(col_rank)
    i2 = decomp%y1en(col_rank)
    js = decomp%x1st(col_rank)
    !$cuf kernel do(3) <<<*, *, stream=a2a_comp>>>
    do k = 1, s3
    do j = i1, i2
    do i = 1, s1
        dst(i+js-1, j-i1+1, k) = src(i, j, k)
    end do
    end do
    end do
    ! Non-self
    do m = 0, dims(1)-1
      if (m == col_rank) cycle
      i1 = decomp%y1st(m)
      i2 = decomp%y1en(m) 
      pos = decomp%y1disp(m) + 1
  !    istat = cudaMemcpy2D( work1_r(pos), s1*(i2-i1+1), src(1,i1,1), s1*s2, s1*(i2-i1+1), s3, cudaMemcpyDeviceToHost )
      istat = cudaMemcpy2D( work1_r_d(pos), s1*(i2-i1+1), src(1,i1,1), s1*s2, s1*(i2-i1+1), s3 )
    end do

    call RGPTLSTOP('----mem_split_yx')
    call RGPTLSTART('----mpi_alltoall_yx')

    select case(backend_type)
#ifdef USE_RDMA
    case(BACKEND_RDMA)
      call rdma_y2x_alltoallv_c()
      call rdma_y2x_alltoallv_wait_c()
#endif

    case(BACKEND_MPI_A2A)
      ! MPI_A2A
      call MPI_ALLTOALLV(work1_r_d, decomp%y1cnts, decomp%y1disp, &
          real_type, work2_r_d, decomp%x1cnts, decomp%x1disp, &
          real_type, DECOMP_2D_COMM_COL, ierror)
    
    case(BACKEND_MPI_P2P)
      ! MPI_P2P
      do iter = 1, dims(1)-1
        sorc = mod(col_rank - iter + dims(1), dims(1))
        dest = mod(col_rank + iter, dims(1))
        call MPI_IRECV( work2_r_d(decomp%x1disp(sorc)+1), decomp%x1cnts(sorc), real_type, &
                        sorc, 0, DECOMP_2D_COMM_COL, reqs_a2a(iter), ierror)
        call MPI_ISEND( work1_r_d(decomp%y1disp(dest)+1), decomp%y1cnts(dest), real_type, &
                        dest, 0, DECOMP_2D_COMM_COL, reqs_a2a(iter+dims(1)-1), ierror)
      end do
      call MPI_WAITALL(2*(dims(1)-1), reqs_a2a, MPI_STATUS_IGNORE, ierror)

#ifdef NCCL
    case(BACKEND_NCCL)
      ! NCCL
      nccl_stat = ncclGroupStart()
      do iter = 1, dims(1)-1
        sorc = mod(col_rank - iter + dims(1), dims(1))
        dest = mod(col_rank + iter, dims(1))
        sorc_glb = sorc * dims(2) + coord(2)
        dest_glb = dest * dims(2) + coord(2)
        nccl_stat = ncclSend(work1_r_d(decomp%y1disp(dest)+1), decomp%y1cnts(dest), nccl_real_type, &
                            dest_glb, nccl_comm, nccl_stream)
        nccl_stat = ncclRecv(work2_r_d(decomp%x1disp(sorc)+1), decomp%x1cnts(sorc), nccl_real_type, &
                            sorc_glb, nccl_comm, nccl_stream)
      end do
      nccl_stat = ncclGroupEnd()
      istat = cudaStreamSynchronize(nccl_stream)
#endif
    end select

    call RGPTLSTOP('----mpi_alltoall_yx')
    call RGPTLSTART('----mem_merge_yx')

    ! rearrange receive buffer
    do m = 0, dims(1)-1
      if (m == col_rank) cycle
      i1 = decomp%x1st(m)
      i2 = decomp%x1en(m)
      pos = decomp%x1disp(m) + 1
  !    istat = cudaMemcpy2D( dst(i1,1,1), d1, work2_r(pos), i2-i1+1, i2-i1+1, d2*d3, cudaMemcpyHostToDevice )
      istat = cudaMemcpy2D( dst(i1,1,1), d1, work2_r_d(pos), i2-i1+1, i2-i1+1, d2*d3 )
    end do

    istat = cudaStreamSynchronize(a2a_comp)

    call RGPTLSTOP('----mem_merge_yx')

#endif

#ifdef NVTX
    call nvtxEndRange
#endif

    return
  end subroutine transpose_y_to_x_real_d
#endif

  subroutine transpose_y_to_x_real(src, dst, opt_decomp)

    implicit none
    
    real(mytype), dimension(:,:,:), intent(IN) :: src
    real(mytype), dimension(:,:,:), intent(OUT) :: dst
    TYPE(DECOMP_INFO), intent(IN), optional :: opt_decomp

    TYPE(DECOMP_INFO) :: decomp
    
    integer :: s1,s2,s3,d1,d2,d3
    integer :: ierror, m, sorc, dest

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

    call RGPTLSTART('----mem_split_yx')

    ! rearrange source array as send buffer
    ! Self-copy
    block
      integer :: i, j, k, i1, i2, js
      i1 = decomp%y1st(col_rank)
      i2 = decomp%y1en(col_rank)
      js = decomp%x1st(col_rank)
      !$OMP PARALLEL DO COLLAPSE(2)
      do k = 1, s3
      do j = i1, i2
      do i = 1, s1
          dst(i+js-1, j-i1+1, k) = src(i, j, k)
      end do
      end do
      end do
    end block
    ! Non-self
    call mem_split_yx_real(src, s1, s2, s3, work1_r, dims(1), col_rank, &
         decomp%y1dist, decomp)

    call RGPTLSTOP('----mem_split_yx')
    call RGPTLSTART('----mpi_alltoall_yx')

    select case(backend_type)
#ifdef USE_RDMA
    case(BACKEND_RDMA)
      call rdma_y2x_alltoallv_c()
      call rdma_y2x_alltoallv_wait_c()
#endif

    case(BACKEND_MPI_A2A)
      ! transpose using MPI_ALLTOALL(V)
#ifdef EVEN
      call MPI_ALLTOALL(work1_r, decomp%y1count, &
           real_type, work2_r, decomp%x1count, &
           real_type, DECOMP_2D_COMM_COL, ierror)
#else
      call MPI_ALLTOALLV(work1_r, decomp%y1cnts, decomp%y1disp, &
           real_type, work2_r, decomp%x1cnts, decomp%x1disp, &
           real_type, DECOMP_2D_COMM_COL, ierror)
#endif
    
    case(BACKEND_MPI_P2P)
      ! MPI_P2P
      do m = 1, dims(1)-1
        sorc = mod(col_rank - m + dims(1), dims(1))
        dest = mod(col_rank + m, dims(1))
        call MPI_IRECV( work2_r(decomp%x1disp(sorc)+1), decomp%x1cnts(sorc), real_type, &
                        sorc, 0, DECOMP_2D_COMM_COL, reqs_a2a(m), ierror)
        call MPI_ISEND( work1_r(decomp%y1disp(dest)+1), decomp%y1cnts(dest), real_type, &
                        dest, 0, DECOMP_2D_COMM_COL, reqs_a2a(m+dims(1)-1), ierror)
      end do
      call MPI_WAITALL(2*(dims(1)-1), reqs_a2a, MPI_STATUS_IGNORE, ierror)

    end select

    call RGPTLSTOP('----mpi_alltoall_yx')
    call RGPTLSTART('----mem_merge_yx')

    ! rearrange receive buffer
    call mem_merge_yx_real(work2_r, d1, d2, d3, dst, dims(1), col_rank, &
         decomp%x1dist, decomp)

    call RGPTLSTOP('----mem_merge_yx')
    
    return
  end subroutine transpose_y_to_x_real

#if 0
  subroutine transpose_y_to_x_complex(src, dst, opt_decomp)

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
    call mem_split_yx_complex(src, s1, s2, s3, work1_c, dims(1), &
         decomp%y1dist, decomp)
    
    ! transpose using MPI_ALLTOALL(V)
#ifdef EVEN
    call MPI_ALLTOALL(work1_c, decomp%y1count, &
         complex_type, work2_c, decomp%x1count, &
         complex_type, DECOMP_2D_COMM_COL, ierror)
#else
    call MPI_ALLTOALLV(work1_c, decomp%y1cnts, decomp%y1disp, &
         complex_type, work2_c, decomp%x1cnts, decomp%x1disp, &
         complex_type, DECOMP_2D_COMM_COL, ierror)
#endif

    ! rearrange receive buffer
    call mem_merge_yx_complex(work2_c, d1, d2, d3, dst, dims(1), &
         decomp%x1dist, decomp)

    return
  end subroutine transpose_y_to_x_complex
#endif

  ! pack/unpack ALLTOALL(V) buffers

  subroutine mem_split_yx_real(in,n1,n2,n3,out,nproc,iproc,dist,decomp)

    implicit none

    integer, intent(IN) :: n1,n2,n3
    real(mytype), dimension(n1,n2,n3), intent(IN) :: in
    real(mytype), dimension(*), intent(OUT) :: out
    integer, intent(IN) :: nproc, iproc
    integer, dimension(0:nproc-1), intent(IN) :: dist
    TYPE(DECOMP_INFO), intent(IN) :: decomp
    
    integer :: i,j,k, m,i1,i2,pos
    integer :: k1,k2
    integer :: offset = 0
    !$ integer :: tid, nt, ksz

    k1 = 1
    k2 = n3
    
    !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(m,k,j,i,i1,i2,pos,tid,nt,k1,k2,ksz,offset)
    !$ tid = OMP_GET_THREAD_NUM()
    !$ nt  = OMP_GET_NUM_THREADS()
    !$ call distribute_among_threads('f', nt, tid, n3, 1, ksz, k1, k2)
    do m=0,nproc-1
       if (m==0) then 
          i1 = 1
          i2 = dist(0)
       else
          i1 = i2+1
          i2 = i1+dist(m)-1
       end if

       if (m == iproc) cycle

       !$ offset = dist(m) * n1 * (k1-1)
#ifdef EVEN
       pos = m * decomp%y1count + 1 + offset
#else
       pos = decomp%y1disp(m) + 1 + offset
#endif

       do k=k1,k2
          do j=i1,i2
             do i=1,n1
                out(pos) = in(i,j,k)
                pos = pos + 1
             end do
          end do
       end do
    end do
    !$OMP END PARALLEL

    return
  end subroutine mem_split_yx_real

#if 0
  subroutine mem_split_yx_complex(in,n1,n2,n3,out,iproc,dist,decomp)

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
       pos = m * decomp%y1count + 1
#else
       pos = decomp%y1disp(m) + 1
#endif

       do k=1,n3
          do j=i1,i2
             do i=1,n1
                out(pos) = in(i,j,k)
                pos = pos + 1
             end do
          end do
       end do
    end do

    return
  end subroutine mem_split_yx_complex
#endif

  subroutine mem_merge_yx_real(in,n1,n2,n3,out,nproc,iproc,dist,decomp)

    implicit none
    
    integer, intent(IN) :: n1,n2,n3
    real(mytype), dimension(*), intent(IN) :: in
    real(mytype), dimension(n1,n2,n3), intent(OUT) :: out
    integer, intent(IN) :: nproc, iproc
    integer, dimension(0:nproc-1), intent(IN) :: dist
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
    do m=0,nproc-1
       if (m==0) then
          i1 = 1
          i2 = dist(0)
       else
          i1 = i2+1
          i2 = i1+dist(m)-1
       end if

       if (m == iproc) cycle

       !$ offset = dist(m) * n2 * (k1-1)
#ifdef EVEN
       pos = m * decomp%x1count + 1 + offset
#else
       pos = decomp%x1disp(m) + 1 + offset
#endif

       do k=k1,k2
          do j=1,n2
             do i=i1,i2
                out(i,j,k) = in(pos)
                pos = pos + 1
             end do
          end do
       end do
    end do
    !$OMP END PARALLEL

    return
  end subroutine mem_merge_yx_real

#if 0
  subroutine mem_merge_yx_complex(in,n1,n2,n3,out,iproc,dist,decomp)

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
       pos = m * decomp%x1count + 1
#else
       pos = decomp%x1disp(m) + 1
#endif

       do k=1,n3
          do j=1,n2
             do i=i1,i2
                out(i,j,k) = in(pos)
                pos = pos + 1
             end do
          end do
       end do
    end do

    return
  end subroutine mem_merge_yx_complex
#endif
