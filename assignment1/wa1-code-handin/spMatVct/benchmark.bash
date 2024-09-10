futhark dataset --i64-bounds=0:9999 -g [1000000]i64 --f32-bounds=-7.0:7.0 -g [1000000]f32 --i64-bounds=100:100 -g [10000]i64 --f32-bounds=-10.0:10.0 -g [10000]f32 > data.in


echo "Generating ground truth data via spMVmult-seq"
futhark c spMVmult-seq.fut && ./spMVmult-seq < data.in > data.out


echo "Benchmarking spMVmult-seq on c backend"
futhark bench spMVmult-seq.fut --entry-point=onlybench --backend=c

echo "Benchmarking spMVmult-flat on cuda backend"
futhark bench spMVmult-flat.fut --entry-point=onlybench --backend=cuda





