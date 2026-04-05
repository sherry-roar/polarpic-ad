        pi = acos(-1.0_fp_pois)
        !$cuf kernel do(1) <<<*,*>>>
        do i = 1, isz+2, 2
            ii = (i-1)/2
            arg = -pi*ii/(2.*isz)
            sincos(1, ii) = sin(arg)
            sincos(2, ii) = cos(arg)
        enddo
        !$cuf kernel do(3) <<<*,*>>>
        do k = 1, sz(3)
        do j = 1, sz(2)
        do i = 1, isz+2, 2
            ii = (i-1)/2
            !work_tmp(ii   ,j,k) =    real( &
            !                         2.*exp(-ri_unit*pi*ii/(2.*isz))*cmplx(cwork(i,j,k),cwork(i+1,j,k),fp) &
            !                        )
            !work_tmp(isz-ii,j,k) = - aimag( &
            !                         2.*exp(-ri_unit*pi*ii/(2.*isz))*cmplx(cwork(i,j,k),cwork(i+1,j,k),fp) &
            !                        ) ! = 0 for ii=0
            !arg = -pi*ii/(2.*isz)
            carg = sincos(2, ii)!cos(arg)
            sarg = sincos(1, ii)!sin(arg)
            work_tmp(ii    , j, k) =  2.*(carg*cwork(i, j, k) - sarg*cwork(i+1, j, k))
            work_tmp(isz-ii, j, k) = -2.*(sarg*cwork(i, j, k) + carg*cwork(i+1, j, k))
        enddo
        enddo
        enddo
        !$cuf kernel do(3) <<<*,*>>>
        do k = 1, sz(3)
        do j = 1, sz(2)
        do i = 1, sz(1)
            work(i, j, k) = work_tmp(i-1, j, k)
        enddo
        enddo
        enddo