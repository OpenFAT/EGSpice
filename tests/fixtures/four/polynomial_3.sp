* Known nonlinear Fourier harmonics
* Expected results: polynomial_3.expected.json
Vin in 0 SIN(0 1 1k)
* Explicit multiplication is a signed cube; ngspice's ^ takes abs(base).
Bout out 0 V=V(in)*V(in)*V(in)
R1 out 0 1k
.tran .2u 5m 0 .2u
.four 1k v(out)
.end
