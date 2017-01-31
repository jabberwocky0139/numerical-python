
# setup
set palette model RGB rgb 33,13,10
unset colorbox

set logscale x
set xr [1e-3:5e-1]
set yr [0:0.5]
# set format y '10^{%L}'
set format x '10^{%L}'

set xlabel '$T/T_c$'
set ylabel '$\Delta Q$'
set key left top

# Q2
# plot for[i=1:3] sprintf('output_g1e-%d.txt', i) using 1:4 w l dt(10, 5) lw 3 palette frac (5-i)/4.0
plot sprintf('output_g1e-%d.txt', 4) using 2:(sqrt($4)) w l lw 3 palette frac (5-1)/4.0 title '$a_s/a_{osc} &=& 0.0001$'
replot sprintf('output_g1e-%d.txt', 3) using 2:(sqrt($4)) w l lw 3 palette frac (5-2)/4.0 title '$a_s/a_{osc} &=& 0.001$'
replot sprintf('output_g1e-%d.txt', 2) using 2:(sqrt($4)) w l lw 3 palette frac (5-3)/4.0 title '$a_s/a_{osc} &=& 0.01$'
replot sprintf('output_g1e-%d.txt', 1) using 2:(sqrt($4)) w l lw 3 palette frac (5-4)/4.0 title '$a_s/a_{osc} &=& 0.1$'

