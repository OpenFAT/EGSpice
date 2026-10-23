* Exact distributed RC line, from the DC limit to strong attenuation.
* Both negative ports are on a biased reference node; AC drives the differential port.
* Oracle: telegrapher ABCD matrix, independently cross-checked with ngspice 45.
Vref ref 0 DC 2
Vin in ref DC 1 AC 1
Rs in a 50
O1 a ref b ref rc
RL b ref 75
.model rc LTRA(r=1000 l=0 g=0 c=100p len=0.1)
.op
.ac dec 1 1 1e12
.end
