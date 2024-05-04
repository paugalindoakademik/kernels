#!/bin/sh

compile()
{
	echo "COMPILING $PWD WITH $1"
	make $1 DATATYPE=FLOAT BLOCKSIZE=4
	make $1 DATATYPE=FLOAT BLOCKSIZE=8
	make $1 DATATYPE=FLOAT BLOCKSIZE=16
	make $1 DATATYPE=FLOAT BLOCKSIZE=32
}

echo "Start"

cd cifar_10
make clean
./CLHT.sh
compile all-bin
cd ..

cd cifar_10_multiple
make clean
./CLHT.sh
compile all-bin
cd ..

cd convolution_2D_bench
make clean
./CLHT.sh
compile all-bin
cd ..

cd correlation_2D
make clean
./CLHT.sh
compile opencl
compile all-hip
cd ..

cd fast_fouriesr_transform_bench
make clean
./CLHT.sh
compile all-bin
cd ..

cd fast_fourier_transform_window_bench
make clean
./CLHT.sh
compile all-bin
cd ..

cd finite_impulse_response_filter
make clean
compile all-bin
cd ..

cd matrix_multiplication_bench
make clean
./CLHT.sh
compile all-bin
cd ..

cd matrix_multiplication_bench_fp16
make clean
compile all-bin
cd ..

cd matrix_multiplication_tensor_bench
make clean
./CLHT.sh
compile all-bin
cd ..

cd max_pooling_bench
make clean
./CLHT.sh
compile all-bin
cd ..

cd relu_bench
make clean
./CLHT.sh
compile all-bin
cd ..

cd softmax_bench
make clean
./CLHT.sh
compile all-bin
cd ..

cd wavelet_transform
make clean
./CLHT.sh
compile opencl
compile all-hip
cd ..

echo "End"
