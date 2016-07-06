unset multiplot; reset

set size ratio 6/12
set key outside

set xlabel 'gN'
set xrange [0:12]

set ylabel 'MAX(Im)'
set yrange [0:0.07]


plot "BdG.txt" w l
