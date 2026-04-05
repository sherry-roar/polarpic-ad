        pi = acos(-1.0_fp_pois)
        !$cuf kernel do(2) <<<*,*>>>
        do k = 1, sz(3)
        do j = 1, sz(2)
            cwork(isz+1,j,k) = 0.0_fp_pois
            cwork(isz+2,j,k) = 0.0_fp_pois
        enddo
        enddo
        !$cuf kernel do(1) <<<*,*>>>
        do i = 1, isz+2, 2
            ii = (i-1)/2
            arg = pi*ii/(2.*isz)
            sincos(1,ii) = sin(arg)
            sincos(2,ii) = cos(arg)
        enddo
        !$cuf kernel do(3) <<<*,*>>>
        do k = 1, sz(3)
        do j = 1, sz(2)
        do i = 1, isz+2, 2
            ii = (i-1)/2
            !work_tmp(2*ii  ,j,k)  = real( 1.*exp(ri_unit*pi*ii/(2.*isz))*(cwork(ii+1,j,k)-ri_unit*cwork(isz-ii+1,j,k)))
            !work_tmp(2*ii+1,j,k)  = aimag(1.*exp(ri_unit*pi*ii/(2.*isz))*(cwork(ii+1,j,k)-ri_unit*cwork(isz-ii+1,j,k)))
            !arg = pi*ii/(2.*isz)
            carg = sincos(2,ii)!cos(arg)
            sarg = sincos(1,ii)!sin(arg)
            work_tmp(2*ii  ,j,k) = 1.*(carg*cwork(ii+1,j,k) + sarg*cwork(isz-ii+1,j,k))
            work_tmp(2*ii+1,j,k) = 1.*(sarg*cwork(ii+1,j,k) - carg*cwork(isz-ii+1,j,k))
        enddo
        enddo
        enddo
        !$cuf kernel do(3) <<<*,*>>>
        do k = 1, sz(3)
        do j = 1, sz(2)
        do i = 1, isz+2
            cwork(i, j, k) = work_tmp(i-1, j, k)
        enddo
        enddo
        enddo