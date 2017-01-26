
# setup
set palette model RGB rgb 33,13,10
unset colorbox

set logscale xy
set xr [1e-3:5e-2]
set yr [5e-5:1e-2]
set format y '10^{%L}'
set format x '10^{%L}'

set xlabel '$T/T_c$'
set ylabel 'Specific heat'
set key right bottom

# Bogoliubov approximation
# plot for[i=1:4] sprintf('specific_g1e-%d.txt', i) using 1:4 w l dt(10, 5) lw 3 palette frac (5-i)/4.0 notitle


# With unperturbed zeromode
# replot for[i=1:4] sprintf('specific_g1e-%d.txt', i) using 1:2 w l lw 3 palette frac (5-i)/4.0 title sprintf('$g = 1\times10^{-%d}$', i)
plot sprintf('specific_g1e-%d.txt', 4) using 1:2 w l dt(10, 5) lw 3 palette frac (5-4)/4.0 notitle
replot sprintf('specific_g5e-%d.txt', 4) using 1:2 w l dt(10, 5) lw 3 lc rgb 'gray' notitle
replot sprintf('specific_g1e-%d.txt', 3) using 1:2 w l dt(10, 5) lw 3 palette frac (5-3)/4.0 notitle


# With perturbed zeromode
# replot 'specific_g1e-4.txt' using 1:3 w l lw 2, 'specific_g5e-4.txt' using 1:3 w l lw 2, 'specific_g1e-3.txt' using 1:3 w l lw 2
replot sprintf('specific_g1e-%d.txt', 4) using 1:3 w l lw 3 palette frac (5-4)/4.0 title sprintf('$g = 1\times10^{-%d}$', 4)
replot sprintf('specific_g5e-%d.txt', 4) using 1:3 w l lw 3 lc rgb 'gray' title sprintf('$g = 5\times 10^{-%d}$', 4)
replot sprintf('specific_g1e-%d.txt', 3) using 1:3 w l lw 3 palette frac (5-3)/4.0 title sprintf('$g = 1\times10^{-%d}$', 3)