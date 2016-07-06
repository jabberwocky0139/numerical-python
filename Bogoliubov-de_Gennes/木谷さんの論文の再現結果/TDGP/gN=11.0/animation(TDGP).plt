unset multiplot; reset

set xrange [-10:10]
set yrange [0:0.8]

do for[i=0:2000000:1000]{
set title sprintf("t=%d",i)
plot sprintf("TDGP%d.txt",i) w l ti "",0.8*sin(x/5)*sin(x/5) ti "Potential"#,"GP.txt" w l
pause 0.1
}


