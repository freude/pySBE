! Created by  on 6/2/21.
subroutine oscillators(t, l_t, k, l_k, omega, ne, nh, mu, damp, V, pulse_d, pulse_w, pulse_a, e_phot, p, pp, ne_k, nh_k)

    ! Compile this as f2py3 -c P_loop.f90 -m P_loop
    implicit none

    ! --------------- input variables -----------------
    integer, parameter :: dp = 8
    integer, parameter :: dim = 3                         ! dimensions
    integer, intent(in) :: l_t, l_k                    ! number of points
    real(dp), intent(in) :: t(l_t)                     ! time array
    real(dp), intent(in) :: k(l_k)                     ! wave vector array
    real(dp), intent(in) :: omega(l_k)                 ! frequencies
    real(dp), intent(in) :: ne(l_k), nh(l_k)           ! distribution functions
    complex(dp), intent(in) :: mu(l_k)                 ! dipole matrix elements
    real(dp), intent(in) :: damp                       ! damping
    real(dp), intent(in) :: V(l_k, l_k)                ! interaction Matrix
    real(dp), intent(in) :: pulse_d, pulse_w, pulse_a  ! external pusle parameters
    real(dp), intent(in) :: e_phot                     ! external pusle parameters

    ! --------------- output variables -----------------
    complex(dp), intent(out) :: pp(l_t, l_k)
    complex(dp), intent(out) :: p(l_t)
    real(dp), intent(out) :: ne_k(l_t, l_k)
    real(dp), intent(out) :: nh_k(l_t, l_k)

    ! --------------- local variables -----------------

    real(dp) :: stk                                    ! step of wave-vector array
    real(dp) :: stt                                    ! step of time array
    integer :: j1, j2                                  ! indices
    ! integration temporary quantities
    complex(dp) :: kk1, kk2, kk3, kk4, A(l_k)
    complex(dp) :: mm1, mm2, mm3, mm4
    complex(dp) :: nn1, nn2, nn3, nn4
    complex(dp) :: Ef1, Ef2, Ef3, Ef4

    real(dp), parameter :: h = 1.054e-34
    real(dp), parameter :: e = 1.602e-19
    complex(dp) :: j_cmplx = Cmplx(0.0_dp, -1.0_dp)

    !f2py intent(in) t, k, omega, eg, ne, nh, mu, damp, h, V, pulse_d, pulse_w, pulse_a, e_phot
    !f2py depend(l_t) p
    !f2py intent(out) pp
    !f2py intent(out) ne_k
    !f2py intent(out) nh_k
    !f2py depend(l_t) p
    !f2py depend(l_t, l_k) pp
    !f2py depend(l_t, l_k) ne_k
    !f2py depend(l_t, l_k) nh_k
    !f2py depend(l_t) t
    !f2py depend(l_k) k, omega, ne, nh, mu, V

    if (size(k) == 1) then
        stk = 1
    else
        stk = k(3) - k(2)
    end if

    stt = t(3) - t(2)

    ! initial conditions
    do j1 = 1, l_k
        ne_k(1, j1) = ne(j1)
        nh_k(1, j1) = nh(j1)
    end do

    do j2 = 2, l_t
        do j1 = 1, l_k

            A(j1) = Sum(V(j1, :) * pp(j2 - 1, :)) * stk

            !-------------------- external field ---------------------

            Ef1 = pulse_a * Exp(-((t(j2 - 1) - pulse_d) ** 2) / (2 * pulse_w ** 2)) * &
                    Exp(j_cmplx * e_phot * t(j2 - 1))
            Ef2 = pulse_a * Exp(-((t(j2 - 1) + stt / 2.0_dp - pulse_d) ** 2) / (2 * pulse_w ** 2)) * &
                    Exp(j_cmplx * e_phot * (t(j2 - 1) + stt / 2.0_dp))
            Ef3 = Ef2
            Ef4 = pulse_a * Exp(-((t(j2 - 1) + stt - pulse_d) ** 2) / (2 * pulse_w ** 2)) * &
                    Exp(j_cmplx * e_phot * (t(j2 - 1) + stt))

            !---------------  Runge-Kutta expressions ----------------

            kk1 = j_cmplx * omega(j1) * pp(j2 - 1, j1) &
                    + j_cmplx * (ne_k(j2 - 1, j1) + nh_k(j2 - 1, j1) - 1.0) * (mu(j1) * Ef1 + A(j1)) - damp * pp(j2 - 1, j1)
            mm1 = 2.0 * imag(pp(j2 - 1, j1) * conjg((mu(j1) * Ef1 + A(j1)) / h))
            nn1 = 2.0 * imag(pp(j2 - 1, j1) * conjg((mu(j1) * Ef1 + A(j1)) / h))

            kk2 = j_cmplx * omega(j1) * (pp(j2 - 1, j1) + stt * kk1 / 2.0_dp)&
                    + j_cmplx * ((ne_k(j2 - 1, j1) + 0.5 * mm1) + (nh_k(j2 - 1, j1) + 0.5 * nn1) - 1.0) &
                            * (mu(j1) * Ef2 + A(j1)) / h &
                    - Damp * (pp(j2 - 1, j1) + stt * kk1 / 2.0_dp)
            mm2 = 2.0 * imag((pp(j2 - 1, j1) + stt * kk1 / 2.0_dp) * conjg((mu(j1) * Ef2 + A(j1)) / h))
            nn2 = 2.0 * imag((pp(j2 - 1, j1) + stt * kk1 / 2.0_dp) * conjg((mu(j1) * Ef2 + A(j1)) / h))

            kk3 = j_cmplx * omega(j1) * (pp(j2 - 1, j1) + stt * kk2 / 2.0_dp)&
                    + j_cmplx * ((ne_k(j2 - 1, j1) + 0.5 * mm2) + (nh_k(j2 - 1, j1) + 0.5 * nn2) - 1.0) &
                            * (mu(j1) * Ef3 + A(j1)) / h &
                    - damp * (pp(j2 - 1, j1) + stt * kk2 / 2.0_dp)
            mm3 = 2.0 * imag((pp(j2 - 1, j1) + stt * kk2 / 2.0_dp) * conjg((mu(j1) * Ef3 + A(j1)) / h))
            nn3 = 2.0 * imag((pp(j2 - 1, j1) + stt * kk2 / 2.0_dp) * conjg((mu(j1) * Ef3 + A(j1)) / h))

            kk4 = j_cmplx * omega(j1) * (pp(j2 - 1, j1) + stt * kk3)&
                    + j_cmplx * ((ne_k(j2 - 1, j1) + mm3) + (nh_k(j2 - 1, j1) + nn3) - 1.0) &
                            * (mu(j1) * Ef4 + A(j1)) / h &
                    - damp * (pp(j2 - 1, j1) + stt * kk3)
            mm4 = 2.0 * imag((pp(j2 - 1, j1) + stt * kk3) * conjg((mu(j1) * Ef4 + A(j1)) / h))
            nn4 = 2.0 * imag((pp(j2 - 1, j1) + stt * kk3) * conjg((mu(j1) * Ef4 + A(j1)) / h))

            !-------------------  polarization -----------------------

            pp(j2, j1) = pp(j2 - 1, j1) + (stt / 6.0_dp) * (kk1 + 2.0_dp * kk2 + 2.0_dp * kk3 + kk4)

            !------------------- electron density --------------------

            ne_k(j2, j1) = ne_k(j2 - 1, j1) + (stt / 6.0_dp) * (mm1 + 2.0_dp * mm2 + 2.0_dp * mm3 + mm4)

            !---------------------- hole density ---------------------

            nh_k(j2, j1) = nh_k(j2 - 1, j1) + (stt / 6.0_dp) * (nn1 + 2.0_dp * nn2 + 2.0_dp * nn3 + nn4)

            !------------------ macroscopic polarization -------------

            if (dim == 2) then
                p(j2) = p(j2) + 1.0 / (2 * 3.1416) * mu(j1) * k(j1) * pp(j2,j1) * stk
            else if (dim == 3) then
                p(j2) = p(j2) + 1.0 / (2 * 3.1416 ** 2) * mu(j1) * k(j1) * k(j1) * pp(j2, j1) * stk
            else if (dim == 1) then
                p(j2) = p(j2) + 1.0 / 3.1416 * mu(j1) * pp(j2, j1) * stk
            else
                p(j2) = 0
            end if

        end do
    end do
    return
end Subroutine oscillators


subroutine polarization(t, l_t, k, l_k, omega, ne, nh, mu, damp, V, pulse_d, pulse_w, pulse_a, e_phot, p)
    ! Compile this as f2py3 -c P_loop.f90 -m P_loop
    implicit none

    ! --------------- input variables -----------------
    integer, parameter :: dp = 8

    integer, intent(in) :: l_t, l_k                    ! number of points
    real(dp), intent(in) :: t(l_t)                     ! time array
    real(dp), intent(in) :: k(l_k)                     ! wave vector array
    real(dp), intent(in) :: omega(l_k)                 ! frequencies
    real(dp), intent(in) :: ne(l_k), nh(l_k)           ! distribution functions
    complex(dp), intent(in) :: mu(l_k)                 ! dipole matrix elements
    real(dp), intent(in) :: damp                       ! damping
    real(dp), intent(in) :: V(l_k, l_k)                ! interaction Matrix
    real(dp), intent(in) :: pulse_d, pulse_w, pulse_a  ! external pusle parameters
    real(dp), intent(in) :: e_phot                     ! external pusle parameters

    ! --------------- output variables -----------------
    complex(dp), intent(out) :: p(l_t)

    ! ---------------- local variables -----------------
    complex(dp) :: pp(l_t, l_k)
    real(dp) :: ne_k(l_t, l_k)
    real(dp) :: nh_k(l_t, l_k)

    !f2py intent(in) t, k, omega, eg, ne, nh, mu, damp, h, V, pulse_d, pulse_w, pulse_a, e_phot
    !f2py intent(out) pp
    !f2py depend(l_t) t
    !f2py depend(l_k) k, omega, ne, nh, mu, V

    call oscillators(t, l_t, k, l_k, omega, ne, nh, mu, damp, V, pulse_d, pulse_w, pulse_a, e_phot, p, pp, ne_k, nh_k)

    return
end subroutine polarization

