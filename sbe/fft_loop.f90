Subroutine loop(E_ft,l_t, l_f, stt,t, fff, P,eps_0,eps, Eg, h, c, n_reff)
  ! Compile this as f2py3 -c fft_loop.f90 -m fft_loop

 
    Implicit None

    ! ==============  Input Variables  ===============
    Integer, Parameter :: DP = 8

    Integer, Intent(in) :: l_t, l_f

    Real(DP), Intent(in) :: stt
    !! step of time array

    Complex(DP), Intent(in) :: E_ft(l_t)
    !! Electric Field

    Complex(DP), Intent(in) :: P(l_t)
    !! Polarization
    Real(DP), Intent(in) :: fff(l_f), t(l_t)
    !! Freq Array

    Real(DP) , Intent(in) :: eps_0, eps, Eg, h, c, n_reff

    !================= Output Variables ==================
    ! The required arrays are written down on to the disk in unformatted form
    ! and read into python for plotting.
    ! ================== to overcome "Dimension" Error ================
    ! Use depend intent specifications
    !f2py depend(l_t) E_ft, P, t
    !f2py depend(l_f) fff

    !! Integration temporary quantities
    Complex(DP) :: ES(l_f), PS(l_f)
    Real(DP) :: PSr(l_f)
    Real(DP) :: pi = 3.14159

    !! ========== Local Variables ==========
    Integer :: j1, j
    !! Looping indexes

    !! Initialization
    ES = 0.0
    PS = 0.0
    PSr = 0.0 
    
    Do j = 1, l_f
       Do j1 = 1, l_t
          ES(j) = ES(j) + E_ft(j1) * Exp(Cmplx(0.0,1.0_DP) * fff(j) * t(j1)) * stt
          PS(j) = PS(j) + P(j1) * Exp(Cmplx(0.0,1.0_DP) * fff(j) * t(j1)) * stt / (4.0 * pi * eps_0 * eps)
       End Do
       PSr(j) = 4 * pi * (fff(j) + Eg / h) * imag(PS(j) / ES(j)) / (c * n_reff)
    End Do


    ! Write the required arrays to drive
    ! The real and imaginary parts are written separately.
    ! This makes reading in python easier.

    Open(unit = 30, file = 'ES_real', form = "unformatted", access = 'stream', status = 'unknown')
    Write(30) Real(ES)
    Close(30)

    Open(unit = 31, file = 'ES_imag', form = "unformatted", access = 'stream', status = 'unknown')
    Write(31) Imag(ES)
    Close(31)

    Open(unit = 20, file = 'PS_real', form = "unformatted", access = 'stream', status = 'unknown')
    Write(20) real(PS)
    Close(20)

    Open(unit = 21, file = 'PS_imag', form = "unformatted", access = 'stream', status = 'unknown')
    Write(21) imag(PS)
    Close(21)

    Open(unit = 40, file = 'PSr', form = "unformatted", access = 'stream', status = 'unknown')
    Write(40) PSr
    Close(40)


    return
End Subroutine loop

