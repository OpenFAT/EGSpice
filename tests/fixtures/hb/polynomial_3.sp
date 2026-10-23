* Nonlinear harmonic generation without device-model ambiguity
* Expected results: polynomial_3.expected.json
Vin in 0 SIN(0 1 1k)
* Explicit multiplication is a signed cube; ngspice's ^ takes abs(base).
Bout out 0 V=V(in)*V(in)*V(in)
Rload out 0 1k
.hb 1k 8
.end
