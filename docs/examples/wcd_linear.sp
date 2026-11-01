Worst-case distance of a margin linear in two Gaussians
* v(out) = 1 + 0.6 z1 + 0.8 z2 crosses 6 at distance 5/|(0.6, 0.8)| = 5
* sigmas, at z = (3, 4).
.param z1=agauss(0,1,1) z2=agauss(0,1,1)
v1 in 0 1
r0 in 0 1k
v2 out 0 {1+0.6*z1+0.8*z2}
r1 out 0 1k
.dc v1 1 2 1
.meas dc m find v(out) at=1
.wcd m '<= 6'
.end
