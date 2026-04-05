        !$cuf kernel do(3) <<<*,*>>>
        do k = 1, sz(3)
        do j = 1, sz(2)
        do i = 1, isz/2
            ii = i-1
            work_tmp(ii      , j, k) = work(2*ii+1            , j, k)
            work_tmp(ii+isz/2, j, k) = work(2*(isz-(ii+isz/2)), j, k)
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