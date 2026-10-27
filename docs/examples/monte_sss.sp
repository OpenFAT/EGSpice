Scaled-sigma sampling of a 5 sigma margin
* v(out) = 0.6 z1 + 0.8 z2 is a unit Gaussian, so m > 5 fails with
* probability Q(5) = 2.87e-7: far too rare for 5000 plain trials.
.param z1=agauss(0,1,1) z2=agauss(0,1,1)
v1 in 0 1
r0 in 0 1k
b1 out 0 v=0.6*z1+0.8*z2
r1 out 0 1k
.dc v1 1 2 1 sweep monte=1000 sigma_scale=2.5,3,3.5,4,4.5 fail='m > 5'
.meas dc m find v(out) at=1
.end
