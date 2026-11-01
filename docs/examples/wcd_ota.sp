5T OTA in unity gain: worst-case distance of the output offset
* Each transistor's vto has a 2 mV local mismatch. The output should stay
* within 10 mV of the 0.9 V input; .dcmatch gives its first-order sigma.
.model nch nmos level=1 vto=0.5 kp=200u lambda=0.05
.model pch pmos level=1 vto=-0.5 kp=100u lambda=0.05
vdd vdd 0 1.8
vin inp 0 0.9
vb bias 0 0.7
m1 d1 inp tail 0 nch w=10u l=1u
m2 out out tail 0 nch w=10u l=1u
m3 d1 d1 vdd vdd pch w=10u l=1u
m4 out d1 vdd vdd pch w=10u l=1u
m5 tail bias 0 0 nch w=10u l=1u
.variation
.local_variation
nmos nch vto=2m
pmos pch vto=2m
.end_local_variation
.end_variation
.dc vdd 1.8 1.9 0.1
.dcmatch v(out)
.meas dc vo find v(out) at=1.8
.meas dc vo2 find v(out) at=1.8
.meas dcmatch s3 find total_3sigma
.wcd vo '<= 0.91' vo2 '>= 0.89' top=3
.end
