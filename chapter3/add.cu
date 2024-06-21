#include <math.h>
#include <stdio.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

// __global__修饰的函数称为核函数，一般由主机调用，在设备中执行
void __global__ add(const double *x, const double *y, double *z);
// __device__修饰的函数称为设备函数，只能被核函数或其他设备函数调用，在设备中执行
// __host__修饰的函数就是主机端的普通C++函数，在主机中被调用，在主机中执行；
// 有时可以用__host__和__device__同时修饰一个函数
void check(const double *z, const int N);

int main(void){
    const int N = 100000000;
    const int M = sizeof(double) * N;
    double *h_x = (double*) malloc(M);
    double *h_y = (double*) malloc(M);
    double *h_z = (double*) malloc(M);
    for (int n = 0; n < N; ++n){
        h_x[n] = a;
        h_y[n] = b;
    }

    // 定义3个双精度类型变量的指针，用`d_`作为所有设备变量的前缀，`h_`作为对应主机变量的前缀
    double *d_x, *d_y, *d_z;
    // 动态分配设备内存，原型： cudaError_t cudaMalloc(void **address, size_t size);
    // 其中，address是待分配设备内存的指针，即指针的指针（双重指针）；size是待分配内存的字节数
    // d_x是一个double类型的指针，它的地址&d_x是double类型的双重指针
    // (void**)是一个强制类型转换操作，将一个某种类型的双重指针转换为一个void类型的双重指针
    // cudaMalloc函数的功能是改变指针d_x本身的值（将一个指针赋值给d_x）
    // 因为C++不支持多个返回值，同时需要改变d_x的值，所以需要传入d_x的地址
    cudaMalloc((void **)&d_x, M);
    cudaMalloc((void **)&d_y, M);
    cudaMalloc((void **)&d_z, M);

    // 将主机中存放在h_x和h_y中的数据复制到设备中的相应变量d_x和d_y所指向的缓冲区中
    // 原型：cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);
    // 其中，dst是目标地址；src是源地址；count是要复制的字节数；kind是枚举类型的变量，表示数据传递的方向，可取以下值：
    // cudaMemcpyHostToHost：主机到主机；cudaMemcpyHostToDevice：主机到设备；
    // cudaMemcpyDeviceToHost：设备到主机；cudaMemcpyDeviceToDevice：设备到设备;
    // cudaMemcpyDefault：根据数据类型自动选择方向
    cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);

    const int block_size = 128;
    const int grid_size = N / block_size;

    add<<<grid_size, block_size>>>(d_x, d_y, d_z);

    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
    
    check(h_z, N);

    free(h_x);
    free(h_y);
    free(h_z);

    // cudaMalloc函数分配的内存需要通过cudaFree函数释放
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    return 0;
}

void __global__ add(const double *x, const double *y, double *z){
    // 在设备的核函数中，用“单指令-多线程”的方式编写代码，故可去掉循环，只需将数组元素指标与线程指标一一对应即可
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N){
        z[n] = x[n] + y[n];
    }
}

// 版本一：有返回值的设备函数
double __device__ add1_device(const double x, const double y){
    return (x + y);
}
void __global__ add1(const double *x, const double *y, double *z, const int N){
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N){
        z[n] = add1_device(x[n], y[n]);
    }
}

// 版本二：用指针的设备函数
void  __device__ add2_device(const double x, const double y, double *z){
    *z = x + y;
}
void __global__ add2(const double *x, ocnst double *y, double *z, const in N){
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N){
        add2_device(x[n], y[n], &z[n]);
    }
}

// 版本三：用引用(reference)的设备函数
void __device__ add3_device(const double x, const double y, double &z){
    z = x + y;
}
void __global__ add3(const double *x, const double *y, double *z, const int N){
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N){
        add3_device(x[n], y[n], z[n]);
    }
}

void check(const double *z, const int N){
    bool has_error = false;
    for (int n = 0; n < N; ++n){
        if (fabs(z[n] - c) > EPSILON){
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}