Subroutine loop(dim, l_t, l_k, t, k, stt, stk, omega, PEg, exce, ne, nh, &
        mu, damp, h, V, pulse_d, pulse_w, pulse_a, e_phot, P, pp, ne_k, nh_k)
  ! Compile this as f2py3 -c P_loop.f90 -m P_loop
  ! This is written in such a way that the integration is interwoven.
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
    Complex(DP), Intent(in) :: mu(l_k)
    !! Dipole Elements

    Real(DP), Intent(in) :: damp
    !! Damping Coefficient
    Real(DP), Intent(in) :: h
    !! Planck's Constant
    Real(DP), Intent(in) :: V(l_k, l_k)
    !! Interaction Matrix

    Real(DP), Intent(in) :: pulse_d, pulse_w, pulse_a, e_phot
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
    Complex(DP) :: mm1, mm2, mm3, mm4
    Complex(DP) :: nn1, nn2, nn3, nn4
    !! Integration temporary quantities
    Complex(DP) :: Ef1, Ef2, Ef3, Ef4
    Complex(DP) :: pp(l_t, l_k)
    Real(DP) :: ne_k(l_t, l_k)
    Real(DP) :: nh_k(l_t, l_k)
    Complex(DP) :: P(l_t)

    !f2py intent(in) dim, l_t, l_k, t, k, stt, stk, omega, PEg, exce, ne, nh, mu, damp, h, V, pulse_d, pulse_w, pulse_a, e_phot
    !f2py intent(out) P
    !f2py intent(out) pp
    !f2py intent(out) ne_k
    !f2py intent(out) nh_k
    !f2py depend(l_t) P
    !f2py depend(l_t, l_k) pp
    !f2py depend(l_t, l_k) ne_k
    !f2py depend(l_t, l_k) nh_k


    Do j1 = 1, l_k
        ne_k(1, j1) = ne(j1)
        nh_k(1, j1) = nh(j1)
    End Do

    Do j2 = 2, l_t
        Do j1 = 1, l_k
            A(j1) = Sum(V(j1, :) * pp(j2 - 1, :)) * stk

            Ef1 = pulse_a*Exp(-((t(j2 - 1) - pulse_d) ** 2) / (2 * pulse_w ** 2)) *&
                    Exp(Cmplx(0.0_DP, 1.0_DP)*(e_phot / h) * t(j2 - 1))
            Ef2 = pulse_a*Exp(-((t(j2 - 1) + stt / 2.0_DP - pulse_d) ** 2) / (2 * pulse_w ** 2)) *&
                    Exp(Cmplx(0.0_DP, 1.0_DP)*(e_phot / h) * (t(j2 - 1)+ stt / 2.0_DP ))
            Ef3 = Ef2
            Ef4 = pulse_a*Exp(-((t(j2 - 1) + stt - pulse_d) ** 2) / (2 * pulse_w ** 2)) *&
                    Exp(Cmplx(0.0_DP, 1.0_DP)*(e_phot / h) * (t(j2 - 1) + stt))

            !-------------------  polarization --------------------

            RS = Cmplx(0.0_DP, -1.0_DP) * (omega(j1) - PEg / h - exce(j1) / h) &
                    * pp(j2 - 1, j1) &
                    + Cmplx(0.0_DP, -1.0_DP) * (ne_k(j2-1, j1) + nh_k(j2-1, j1) - 1.0) &
                            * (mu(j1) * Ef1 + A(j1)) / h &
                    - damp * pp(j2 - 1, j1) / h

            kk1 = RS
            mm1 = 2.0 * imag(pp(j2 - 1, j1) * Conjg((mu(j1) * Ef1 + A(j1)) / h))
            nn1 = 2.0 * imag(pp(j2 - 1, j1) * Conjg((mu(j1) * Ef1 + A(j1)) / h))

            kk2 = Cmplx(0.0_DP, -1.0_DP) * (omega(j1) - PEg / h - exce(j1) / h) &
                    * (pp(j2 - 1, j1) + stt * kk1 / 2.0_DP)&
                    + Cmplx(0.0_DP, -1.0_DP) * ( (ne_k(j2-1, j1) + 0.5 * mm1) + (nh_k(j2-1, j1) +0.5*nn1) - 1.0) &
                            * (mu(j1) * Ef2 + A(j1)) / h &
                            - Damp * (pp(j2 - 1, j1) + stt * kk1 / 2.0_DP) / h
            mm2 = 2.0 * imag((pp(j2 - 1, j1) + stt * kk1 / 2.0_DP) * Conjg((mu(j1) * Ef2 + A(j1)) / h))
            nn2 = 2.0 * imag((pp(j2 - 1, j1) + stt * kk1 / 2.0_DP) * conjg((mu(j1) * Ef2 + A(j1)) / h))
            

            kk3 = Cmplx(0.0_DP, -1.0_DP) * (omega(j1) - PEg / h - exce(j1) / h) &
                    * (pp(j2 - 1, j1) + stt * kk2 / 2.0_DP)&
                    + Cmplx(0.0_DP, -1.0_DP) * ((ne_k(j2-1, j1) + 0.5*mm2) + (nh_k(j2-1, j1) +0.5*nn2) - 1.0) &
                            * (mu(j1) * Ef3 + A(j1)) / h &
                    - damp * (pp(j2 - 1, j1) + stt * kk2 / 2.0_DP) / h
            mm3 = 2.0 * imag((pp(j2 - 1, j1) + stt * kk2 / 2.0_DP) * Conjg((mu(j1) * Ef3 + A(j1)) / h))
            nn3 = 2.0 * imag((pp(j2 - 1, j1) + stt * kk2 / 2.0_DP) * conjg((mu(j1) * Ef3 + A(j1)) / h))
            
            kk4 = Cmplx(0.0_DP, -1.0_DP) * (omega(j1) - PEg / h - exce(j1) / h) &
                    * (pp(j2 - 1, j1) + stt * kk3)&
                    + Cmplx(0.0_DP, -1.0_DP) * ((ne_k(j2-1, j1) + mm3) + (nh_k(j2-1, j1)+ nn3) - 1.0) &
                            * (mu(j1) * Ef4 + A(j1)) / h &
                    - damp * (pp(j2 - 1, j1) + stt * kk3) / h
            mm4 = 2.0 * imag((pp(j2 - 1, j1) + stt * kk3) * Conjg((mu(j1) * Ef4 + A(j1)) / h))
            nn4 = 2.0 * imag((pp(j2 - 1, j1) + stt * kk3) * Conjg((mu(j1) * Ef4 + A(j1)) / h))


            pp(j2, j1) = pp(j2 - 1, j1) + (stt / 6.0_DP) * (kk1 + 2.0_DP * kk2 + 2.0_DP * kk3 + kk4)

            if (dim == 2) then
                P(j2) = P(j2) +  1.0 / (2 * 3.1416) * mu(j1) * k(j1) * pp(j2,j1) * stk
            else if (dim == 3) then
                P(j2) = P(j2) + 1.0 / (2 * 3.1416 ** 2) * mu(j1) * k(j1) * k(j1) * pp(j2, j1) * stk
            else if (dim == 1) then
                P(j2) = P(j2) + 1.0 / 3.1416 * mu(j1) * pp(j2, j1) * stk
            else
                P(j2) =  P(j2) + 1.0 / 3.1416 * mu(j1) * pp(j2, j1) * stk * sqrt(k(j1))
            end if

            !------------------- electron density --------------------

            ne_k(j2, j1) = ne_k(j2 - 1, j1) + (stt / 6.0_DP) * (mm1 + 2.0_DP * mm2 + 2.0_DP * mm3 + mm4)

            !---------------------- hole density ---------------------

            nh_k(j2, j1) = nh_k(j2 - 1, j1) + (stt / 6.0_DP) * (nn1 + 2.0_DP * nn2 + 2.0_DP * nn3 + nn4)

        End Do
    End Do
    return
End Subroutine loop

