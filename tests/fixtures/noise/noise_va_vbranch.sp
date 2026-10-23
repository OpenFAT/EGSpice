* Verilog-A noise on a potential contribution: V(p,n) <+ R*I(p,n) + white_noise(Sv)
* Expected results: noise_va_vbranch.expected.json
.hdl "va_noise.assets/noiseless_res.va"
.hdl "va_noise.assets/vseries_res.va"
Vin in 0 DC 0 AC 1
Ns in out noiseless_res
Nv out 0 vseries_res
.noise v(out) Vin lin 2 1000 2000
.end
