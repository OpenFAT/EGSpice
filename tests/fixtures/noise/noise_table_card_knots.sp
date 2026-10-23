* LRM 4.6.4.3 noise_table over an array parameter: the card's knots are the PSD
* Expected results: noise_table_card_knots.expected.json
.hdl "va_noise.assets/noiseless_res.va"
.hdl "va_noise.assets/ntab_card.va"
.model tm ntab_card tbl[1]=2e-18 tbl[3]=18e-18
Vin in 0 DC 0 AC 1
Ns in out noiseless_res
Nt out 0 tm
.noise v(out) Vin lin 2 1000 5000
.end
