
std::string kernel_code = 
"void kernel kernel_matrix_convolution_old(global const bench_t* A,  global bench_t* B, global const bench_t* kernel_data, const int n, const int m, const int w, const int kernel_size ){\n"
"int x = get_global_id(0);\n"
"int y = get_global_id(1);\n"
"unsigned int size = n;\n"
"int kernel_rad = kernel_size / 2;\n"
"bench_t sum = 0;\n"
"if (x < size && y < size){\n"
"for(int i = -kernel_rad; i <= kernel_rad; ++i) // loop over kernel_rad  -1 to 1 in kernel_size 3\n"
"{\n"
"for(int j = -kernel_rad; j <= kernel_rad; ++j)\n"
"{\n"
"bench_t value = 0;\n"
"if (i + x < 0 || j + y < 0)\n"
"{\n"
"value = 0;\n"
"}\n"
"else if ( i + x > size - 1 || j + y > size - 1)\n"
"{\n"
"value = 0;\n"
"}\n"
"else\n"
"{\n"
"value = A[(x + i)*size+(y + j)];\n"
"}\n"
"sum += value * kernel_data[(i+kernel_rad)* kernel_size + (j+kernel_rad)];\n"
"}\n"
"}\n"
"B[x*size+y ] = sum;\n"
"}\n"
"}\n"
"void kernel kernel_matrix_convolution(global const bench_t* A,  global bench_t* B, global const bench_t* kernel_data, const int n, const int m, const int w, const int kernel_size, local bench_t* data, const int shared_size, const int kernel_rad){\n"
"int x = get_global_id(0);\n"
"int y = get_global_id(1);\n"
"unsigned int size = n;\n"
"int x0, y0;\n"
"bench_t sum = 0;\n"
"if (x < size && y < size){\n"
"//TOP right corner\n"
"x0 = x - kernel_rad;\n"
"y0 = y - kernel_rad;\n"
"if ( x0 < 0 || y0 < 0 )\n"
"{\n"
"data[get_local_id(0) * shared_size + get_local_id(1)] = 0;\n"
"}\n"
"else\n"
"{\n"
"data[get_local_id(0) * shared_size + get_local_id(1)] = A[x0 *size+y0];\n"
"}\n"
"//BOTTOM right corner\n"
"x0 = x + kernel_rad;\n"
"y0 = y - kernel_rad;\n"
"if ( x0 > size-1  || y0 < 0 )\n"
"{\n"
"data[(get_local_id(0) + kernel_rad * 2) * shared_size + get_local_id(1)] = 0;\n"
"}\n"
"else\n"
"{\n"
"data[(get_local_id(0) + kernel_rad * 2) * shared_size + get_local_id(1)] = A[x0 *size+y0];\n"
"}\n"
"//TOP left corner\n"
"x0 = x - kernel_rad;\n"
"y0 = y + kernel_rad;\n"
"if ( x0 < 0  || y0 > size-1 )\n"
"{\n"
"data[get_local_id(0) * shared_size + (get_local_id(1) + kernel_rad * 2)] = 0;\n"
"}\n"
"else\n"
"{\n"
"data[get_local_id(0) * shared_size + (get_local_id(1) + kernel_rad * 2)] = A[x0 *size+y0];\n"
"}\n"
"//BOTTOM left corner\n"
"x0 = x + kernel_rad;\n"
"y0 = y + kernel_rad;\n"
"if ( x0 > size-1  || y0 > size-1 )\n"
"{\n"
"data[(get_local_id(0) + kernel_rad * 2) * shared_size + (get_local_id(1) + kernel_rad * 2)] = 0;\n"
"}\n"
"else\n"
"{\n"
"data[(get_local_id(0) + kernel_rad * 2) * shared_size + (get_local_id(1) + kernel_rad * 2)] = A[x0 *size+y0];\n"
"}\n"
"barrier(CLK_LOCAL_MEM_FENCE);\n"
"unsigned int xa = kernel_rad + get_local_id(0);\n"
"unsigned int ya = kernel_rad + get_local_id(1);\n"
"for(int i = -kernel_rad; i <= kernel_rad; ++i)\n"
"{\n"
"for(int j = -kernel_rad; j <= kernel_rad; ++j)\n"
"{\n"
"sum += data[(xa + i) * shared_size +  (ya + j)] * kernel_data[(i+kernel_rad)* kernel_size + (j+kernel_rad)];\n"
"}\n"
"}\n"
"B[x*size+y ] = sum;\n"
"}\n"
"}\n"
"void kernel kernel_relu(global const bench_t* A, global bench_t* B, const int size ){\n"
"int i = get_global_id(0);\n"
"if (i < (size * size) ){\n"
"bench_t threshold = 0;\n"
"B[i] = max(threshold, A[i]);\n"
"}\n"
"}\n"
"void kernel kernel_relu_linear(global const bench_t* A, global bench_t* B, const int size ){\n"
"int i = get_global_id(0);\n"
"if (i < size){\n"
"bench_t threshold = 0;\n"
"B[i] = max(threshold, A[i]);\n"
"}\n"
"}\n"
"void kernel kernel_max(global const bench_t* A, global bench_t* B, const int size, const  int stride,  const  int lateral_stride ){\n"
"int i = get_global_id(0);\n"
"if (i < lateral_stride*lateral_stride){\n"
"bench_t max_value = A[(i * stride + ((i/lateral_stride)*size))];\n"
"for(unsigned int x = 0; x < stride; ++x)\n"
"{\n"
"for(unsigned int y = 0; y < stride; ++y)\n"
"{\n"
"max_value = max(max_value, A[((i * stride + ((i/lateral_stride)*size)) + x)  + ( y * size)]);\n"
"}\n"
"}\n"
"B[i] = max_value;\n"
"}\n"
"}\n"
"void kernel kernel_lrn(global const bench_t* A, global bench_t* B, const int size, const bench_t K, const bench_t ALPHA, const bench_t BETA ){\n"
"int i = get_global_id(0);\n"
"int j = get_global_id(1);\n"
"if (i < size && j < size){\n"
"B[i*size+j] = A[i*size+j]/pow((K+ALPHA*pow(A[i*size+j],2)),BETA);\n"
"}\n"
"}\n"
"void kernel kernel_matrix_multiplication(global const bench_t* A, const global bench_t* B, global bench_t* C, const int n, const int m, const int w ){\n"
"int i = get_global_id(0);\n"
"int j = get_global_id(1);\n"
"if (i < n && j < m){\n"
"bench_t acumulated = 0;\n"
"for (unsigned int k_d = 0; k_d < w; ++k_d )\n"
"{\n"
"acumulated += A[i*w+k_d] * B[k_d*m +j];\n"
"}\n"
"C[i*m+j] =  acumulated;\n"
"}\n"
"}\n"
"void kernel kernel_softmax(global const bench_t* A, global bench_t* B, global bench_t* sum_d_B, const int size ){\n"
"int i = get_global_id(0);\n"
"int tid = get_local_id(0);\n"
"bench_t value = 0;\n"
"__local bench_t shared_data[BLOCK_SIZE];\n"
"if (i < (size) ){\n"
"value = exp(A[i]);\n"
"B[i] = value;\n"
"shared_data[tid] = value;\n"
"barrier(CLK_LOCAL_MEM_FENCE);\n"
"for (unsigned int s=get_local_size(0)/2; s>0; s>>=1)\n"
"{\n"
"if (tid < s)\n"
"{\n"
"shared_data[tid] += shared_data[tid + s];\n"
"}\n"
"}\n"
"barrier(CLK_LOCAL_MEM_FENCE);\n"
"if (tid == 0)\n"
"{\n"
"atomic_add_global(sum_d_B, shared_data[0]);\n"
"}\n"
"}\n"
"}\n"
"void kernel kernel_softmax_end(global  bench_t* B, global bench_t* sum_d_B, const int size){\n"
"int i = get_global_id(0);\n"
"if (i < (size) ){\n"
"B[i] = (B[i]/(*sum_d_B));\n"
"}\n"
"}\n"
;