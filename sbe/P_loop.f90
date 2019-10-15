Subroutine loop(dim, l_t, l_k, t, k, stt, stk, omega, PEg, exce, ne, nh, &
        mu, damp, h, V, pulse_d, pulse_w, pulse_a)
    ! Compile this as f2py3 -c P_loop.f90 -m P_loop
    Implicit None

    ! ==============  Input ===============
    Integer, Parameter :: DP = 8

    ! Exchange Energy

    Integer, Intent(in) :: l_t, l_k, dim
    Real(DP), Intent(in) :: stk
    !! step of wave-vector array
    Real(DP), Intent(in) :: stt
    !! step of time array

    Real(DP), Intent(in) :: t(l_t)
    !! step of wave-vector array
    Real(DP), Intent(in) :: k(l_k)
    !! step of time array

    Real(DP), Intent(in) :: omega(l_k)
    !! Transition Frequencies
    Real(DP), Intent(in) :: PEg
    !! Band_edge Transition frequency
    Real(DP), Intent(in) :: exce(l_k)
    !! Exchange Energy
    Real(DP), intent(in) :: ne(l_k), nh(l_k)
    !! Distribution Functions
    Real(DP), Intent(in) :: mu(l_k)
    !! Dipole Elements

    Real(DP), Intent(in) :: damp
    !! Damping Coefficient
    Real(DP), Intent(in) :: h
    !! Planck's Constant
    Real(DP), Intent(in) :: V(l_k, l_k)
    !! Interaction Matrix

    Real(DP), Intent(in) :: pulse_d, pulse_w, pulse_a
    !! Pusle Delay and Pulse widths
    !! Pulse Parameters

    !================= Output ==================
    ! The required arrays are written down on to the disk in unformatted form
    ! and read into python for plotting.
    ! ================== to overcome "Dimension" Error ================
    ! Use depend intent specifications
    !f2py depend(l_t) t
    !f2py depend(l_k) k, omega, exce, ne, nh, mu, V

    ! ============== Local =================

    !! Local Variables
    Integer :: j1, j2
    !! Looping indexes
    Complex(DP) :: RS, kk1, kk2, kk3, kk4, A(l_k)
    !! Integration temporary quantities
    Real(DP) :: Ef1, Ef2, Ef3, Ef4
    Complex(DP) :: pp(l_t, l_k)
    Complex(DP) :: P(l_t)

    Do j2 = 2, l_t
        Do j1 = 1, l_k
            A(j1) = Sum(V(j1, :) * pp(j2 - 1, :)) * stk

            Ef1 = pulse_a*Exp(-((t(j2 - 1) - pulse_d) ** 2) / (2 * pulse_w ** 2))
            Ef2 = pulse_a*Exp(-((t(j2 - 1) + stt / 2.0_DP - pulse_d) ** 2) / (2 * pulse_w ** 2))
            Ef3 = Ef2
            Ef4 = pulse_a*Exp(-((t(j2 - 1) + stt - pulse_d) ** 2) / (2 * pulse_w ** 2))

            RS = Cmplx(0.0_DP, -1.0_DP) * (omega(j1) - PEg / h - exce(j1) / h) &
                    * pp(j2 - 1, j1) &
                    + Cmplx(0.0_DP, -1.0_DP) * (ne(j1) - nh(j1)) &
                            * (mu(j1) * Ef1 + A(j1)) / h &
                    - damp * pp(j2 - 1, j1) / h

            kk1 = RS

            kk2 = Cmplx(0.0_DP, -1.0_DP) * (omega(j1) - PEg / h - exce(j1) / h) &
                    * (pp(j2 - 1, j1) + stt * kk1 / 2.0_DP)&
                    + Cmplx(0.0_DP, -1.0_DP) * (ne(j1) - nh(j1)) &
                            * (mu(j1) * Ef2 + A(j1)) / h &
                    - Damp * (pp(j2 - 1, j1) + stt * kk1 / 2.0_DP) / h

            kk3 = Cmplx(0.0_DP, -1.0_DP) * (omega(j1) - PEg / h - exce(j1) / h) &
                    * (pp(j2 - 1, j1) + stt * kk2 / 2.0_DP)&
                    + Cmplx(0.0_DP, -1.0_DP) * (ne(j1) - nh(j1)) &
                            * (mu(j1) * Ef3 + A(j1)) / h &
                    - damp * (pp(j2 - 1, j1) + stt * kk2 / 2.0_DP) / h

            kk4 = Cmplx(0.0_DP, -1.0_DP) * (omega(j1) - PEg / h - exce(j1) / h) &
                    * (pp(j2 - 1, j1) + stt * kk3)&
                    + Cmplx(0.0_DP, -1.0_DP) * (ne(j1) - nh(j1)) &
                            * (mu(j1) * Ef4 + A(j1)) / h &
                    - damp * (pp(j2 - 1, j1) + stt * kk3) / h

            !Print *, 'kks ', kk1,kk2,kk3,kk4

            pp(j2, j1) = pp(j2 - 1, j1) + (stt / 6.0_DP) * (kk1 + 2.0_DP * kk2 + 2.0_DP * kk3 + kk4)

            if (dim == 2) then
                P(j2) = P(j2) +  1.0 / (2 * 3.1416) * mu(j1) * k(j1) * pp(j2,j1) * stk
            else
                P(j2) = P(j2) + 1.0 / (2 * 3.1416 ** 2) * mu(j1) * k(j1) * k(j1) * pp(j2, j1) * stk
            end if



        End Do

    End Do


    ! Write the required arrays to drive
    ! The real and imaginary parts are written separately.
    ! This makes reading in python easier.

    Open(unit = 30, file = 'pp_real', form = "unformatted", access = 'stream', status = 'unknown')
    Write(30) Real(pp)
    Close(30)

    Open(unit = 31, file = 'pp_imag', form = "unformatted", access = 'stream', status = 'unknown')
    Write(31) Imag(pp)
    Close(31)

    Open(unit = 20, file = 'P_real', form = "unformatted", access = 'stream', status = 'unknown')
    Write(20) real(P)
    Close(20)

    Open(unit = 21, file = 'P_imag', form = "unformatted", access = 'stream', status = 'unknown')
    Write(21) imag(P)
    Close(21)

    return
End Subroutine loop

