* Independent ngspice 45 BSIM4 AD channel-noise regression: multiplicity3
VDD vdd 0 1.8
VG g 0 DC 0.9 AC 1
VB b 0 0
RL vdd d 5k
M1 d g 0 b n4 W=10u L=0.5u NF=1 M=3
.model n4 NMOS LEVEL=54 TNOIMOD=1 FNOIMOD=0 RDSMOD=1 RSH=10 RDSWMIN=100 RDSW=200 RNOIB=1.2 
.noise v(d) VG dec 3 10 10g
.end
