* ngspice 45 default powers use abs(base), including folded integer exponents.
* PTpower/PTpowerH; explicit multiplication remains a signed cube.
* DC sweep checks both signs and zero; AC checks the derivative at -2 V.
* A variable exponent also exercises the all-zero Newton start (0^0 = 1).
Vin in 0 DC -2 AC 1
Vexp exponent 0 3
Bcaret caret 0 V=V(in)^3
Bstar star 0 V=V(in)**3
Bfunc func 0 V=pow(V(in),3)
Bvariable variable 0 V=V(in)^V(exponent)
Bcube cube 0 V=V(in)*V(in)*V(in)
Bsquare square 0 V=V(in)^2
Bfraction fraction 0 V=V(in)^2.5
Bfold folded 0 V={(-2)^3}
.op
.ac lin 1 1k 1k
.dc Vin -2 2 1
.end
