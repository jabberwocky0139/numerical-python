
# setup
set logscale xy
set xr [1e-3:5e-1]
set yr [1e-6:3e0]
set format y "10^{%L}"
set format x "10^{%L}"


# Bogoliubov approximation
plot 'specific_g_nozeromode0.0001.txt' w l dt(10, 5) lw 2, 'specific_g_nozeromode0.001.txt' w l dt(10, 5) lw 2, 'specific_g_nozeromode0.01.txt' w l dt(10, 5) lw 2, 'specific_g_nozeromode0.1.txt' w l dt(10, 5) lw 2

# With zeromode
replot 'specific_g1e-4.txt' using 1:2 w l lw 2, 'specific_g1e-3.txt' using 1:2 w l lw 2, 'specific_g1e-2.txt' using 1:2 w l lw 2, 'specific_g1e-1.txt' using 1:2 w l lw 2 