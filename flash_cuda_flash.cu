#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define D 64         // Head Dimension

#define CHECK_CUDA(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// Helper: Atomic Add for Half
#if __CUDA_ARCH__ < 700
__device__ void atomicAddHalf(half* address, half val) {
    unsigned int* address_as_ui = (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;
    do {
        assumed = old;
        half h_old = (size_t)address & 2 ? __ushort_as_half(old >> 16) : __ushort_as_half(old & 0xffff);
        half h_sum = __hadd(h_old, val);
        unsigned int next = (size_t)address & 2 
            ? (old & 0xffff) | (__half_as_ushort(h_sum) << 16) 
            : (old & 0xffff0000) | __half_as_ushort(h_sum);
        old = atomicCAS(address_as_ui, assumed, next);
    } while (assumed != old);
}
#else
__device__ void atomicAddHalf(half* address, half val) {
    atomicAdd(address, val);
}
#endif

// Helper Kernels
__global__ void float2half_kernel(half* out, const float* in, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) out[i] = __float2half(in[i]);
}

__global__ void compute_delta_half(const half* dO, const half* O, float* Delta, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = 0.0f;
        for (int x = 0; x < d; x++) sum += __half2float(dO[i * d + x]) * __half2float(O[i * d + x]);
        Delta[i] = sum;
    }
}

// Flash Kernel (Optimized for Half Precision)
template<int BR, int BC>
__global__ void flash_optimized_half(
    const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V, 
    const half* __restrict__ dO, const float* __restrict__ L, const float* __restrict__ m, 
    half* __restrict__ dQ, half* __restrict__ dK, half* __restrict__ dV, 
    const float* __restrict__ Delta, float scale, int n
) {
    __shared__ half sQ[BR * D];
    __shared__ half sdO[BR * D];
    __shared__ half sK[BC * D];
    __shared__ half sV[BC * D];

    int tx = threadIdx.x; int ty = threadIdx.y; 
    int row_q_start = blockIdx.x * BR; int q_idx = row_q_start + ty;

    if (row_q_start < n) {
        for (int x = tx; x < D; x += BC) {
            sQ[ty * D + x] = Q[row_q_start * D + x];
            sdO[ty * D + x] = dO[row_q_start * D + x];
        }
    }
    __syncthreads();

    for (int j_start = 0; j_start < n; j_start += BC) {
        for (int x = tx; x < D; x += BC) {
            sK[ty * D + x] = K[(j_start + ty) * D + x];
            sV[ty * D + x] = V[(j_start + ty) * D + x];
        }
        __syncthreads();

        for (int k_row = 0; k_row < BC; k_row++) {
            int global_k = j_start + k_row;
            float dot = 0.0f;
            for (int x = 0; x < D; x++) dot += __half2float(sQ[ty * D + x]) * __half2float(sK[k_row * D + x]);
            
            float val_f = expf(dot * scale - m[q_idx]) / L[q_idx];
            float dP_val = 0.0f;
            for (int x = 0; x < D; x++) dP_val += __half2float(sdO[ty * D + x]) * __half2float(sV[k_row * D + x]);

            float dS_f = val_f * (dP_val - Delta[q_idx]) * scale;

            for (int x = tx; x < D; x += BC) {
                atomicAddHalf(&dQ[q_idx * D + x], __float2half(dS_f * __half2float(sK[k_row * D + x])));
                atomicAddHalf(&dK[global_k * D + x], __float2half(dS_f * __half2float(sQ[ty * D + x])));
                atomicAddHalf(&dV[global_k * D + x], __float2half(val_f * __half2float(sdO[ty * D + x])));
            }
        }
        __syncthreads();
    }
}

// Runner for cache flushing
__global__ void flush_l2_cache_kernel(float* buffer, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        buffer[idx] += 1.0f;
    }
}

int main() {
    int seq_lengths[] = {2048, 4096, 8192, 16384};
    int num_seqs = 4;

    printf("========================================================================\n");
    printf("| %-8s | %-12s | %-12s | %-12s | %-12s |\n", "Seq Len", "4x4 (ms)", "8x8 (ms)", "16x16 (ms)", "32x32 (ms)");
    printf("========================================================================\n");

    for (int t = 0; t < num_seqs; t++) {
        int n = seq_lengths[t];
        
        // Allocations
        size_t sz_h = (size_t)n * D * sizeof(half);
        half *d_Q, *d_K, *d_V, *d_O, *d_dO, *d_dQ, *d_dK, *d_dV;
        float *d_L, *d_m, *d_Delta;

        CHECK_CUDA(cudaMalloc(&d_Q, sz_h)); CHECK_CUDA(cudaMalloc(&d_K, sz_h)); 
        CHECK_CUDA(cudaMalloc(&d_V, sz_h)); CHECK_CUDA(cudaMalloc(&d_O, sz_h)); 
        CHECK_CUDA(cudaMalloc(&d_dO, sz_h)); 
        CHECK_CUDA(cudaMalloc(&d_dQ, sz_h)); CHECK_CUDA(cudaMalloc(&d_dK, sz_h)); 
        CHECK_CUDA(cudaMalloc(&d_dV, sz_h));
        CHECK_CUDA(cudaMalloc(&d_L, n*4)); CHECK_CUDA(cudaMalloc(&d_m, n*4)); 
        CHECK_CUDA(cudaMalloc(&d_Delta, n*4));

        // Initialization
        float* h_dummy = (float*)malloc(n*D*4);
        for(int i=0; i<n*D; i++) h_dummy[i] = 0.1f;
        float* d_temp; cudaMalloc(&d_temp, n*D*4);
        cudaMemcpy(d_temp, h_dummy, n*D*4, cudaMemcpyHostToDevice);
        float2half_kernel<<<(n*D)/256+1, 256>>>(d_Q, d_temp, n*D);
        float2half_kernel<<<(n*D)/256+1, 256>>>(d_K, d_temp, n*D);
        float2half_kernel<<<(n*D)/256+1, 256>>>(d_V, d_temp, n*D);
        float2half_kernel<<<(n*D)/256+1, 256>>>(d_O, d_temp, n*D);
        float2half_kernel<<<(n*D)/256+1, 256>>>(d_dO, d_temp, n*D);

        // Pre-compute constant terms
        compute_delta_half<<<n/256+1, 256>>>(d_dO, d_O, d_Delta, n, D);

        // Flush Buffer
        int flush_size = 64 * 1024 * 1024;
        float *d_flush; CHECK_CUDA(cudaMalloc(&d_flush, flush_size * sizeof(float)));

        float scale = 1.0f / sqrtf(D);
        cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
        float ms4=0, ms8=0, ms16=0, ms32=0;

        // TEST 0: Block 4x4 
        {
            CHECK_CUDA(cudaMemset(d_dQ, 0, sz_h)); CHECK_CUDA(cudaMemset(d_dK, 0, sz_h)); CHECK_CUDA(cudaMemset(d_dV, 0, sz_h));
            flush_l2_cache_kernel<<<flush_size/256, 256>>>(d_flush, flush_size);
            
            dim3 block(4, 4); 
            dim3 grid(n / 4);
            cudaEventRecord(start);
            flash_optimized_half<4, 4><<<grid, block>>>(d_Q, d_K, d_V, d_dO, d_L, d_m, d_dQ, d_dK, d_dV, d_Delta, scale, n);
            cudaEventRecord(stop); cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms4, start, stop);
        }

        // TEST 1: Block 8x8 ---
        {
            CHECK_CUDA(cudaMemset(d_dQ, 0, sz_h)); CHECK_CUDA(cudaMemset(d_dK, 0, sz_h)); CHECK_CUDA(cudaMemset(d_dV, 0, sz_h));
            flush_l2_cache_kernel<<<flush_size/256, 256>>>(d_flush, flush_size);
            
            dim3 block(8, 8); 
            dim3 grid(n / 8);
            cudaEventRecord(start);
            flash_optimized_half<8, 8><<<grid, block>>>(d_Q, d_K, d_V, d_dO, d_L, d_m, d_dQ, d_dK, d_dV, d_Delta, scale, n);
            cudaEventRecord(stop); cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms8, start, stop);
        }

        // TEST 2: Block 16x16 ---
        {
            CHECK_CUDA(cudaMemset(d_dQ, 0, sz_h)); CHECK_CUDA(cudaMemset(d_dK, 0, sz_h)); CHECK_CUDA(cudaMemset(d_dV, 0, sz_h));
            flush_l2_cache_kernel<<<flush_size/256, 256>>>(d_flush, flush_size);
            
            dim3 block(16, 16); 
            dim3 grid(n / 16);
            cudaEventRecord(start);
            flash_optimized_half<16, 16><<<grid, block>>>(d_Q, d_K, d_V, d_dO, d_L, d_m, d_dQ, d_dK, d_dV, d_Delta, scale, n);
            cudaEventRecord(stop); cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms16, start, stop);
        }

        // TEST 3: Block 32x32 ---
        {
            CHECK_CUDA(cudaMemset(d_dQ, 0, sz_h)); CHECK_CUDA(cudaMemset(d_dK, 0, sz_h)); CHECK_CUDA(cudaMemset(d_dV, 0, sz_h));
            flush_l2_cache_kernel<<<flush_size/256, 256>>>(d_flush, flush_size);
            
            dim3 block(32, 32); 
            dim3 grid(n / 32);
            cudaEventRecord(start);
            flash_optimized_half<32, 32><<<grid, block>>>(d_Q, d_K, d_V, d_dO, d_L, d_m, d_dQ, d_dK, d_dV, d_Delta, scale, n);
            cudaEventRecord(stop); cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms32, start, stop);
        }

        printf("| %-8d | %-12.2f | %-12.2f | %-12.2f | %-12.2f |\n", n, ms4, ms8, ms16, ms32);

        // Free (I had forgotten this)
        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); 
        cudaFree(d_dO); cudaFree(d_dQ); cudaFree(d_dK); cudaFree(d_dV); 
        cudaFree(d_L); cudaFree(d_m); cudaFree(d_Delta); cudaFree(d_flush);
        cudaFree(d_temp); free(h_dummy);
    }
    printf("========================================================================\n");
    return 0;
}
