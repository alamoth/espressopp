#define realG double
#define realG3 double4
#define realG4 double4
#define make_realG3 make_double4
#define make_realG4 make_double4
#ifdef __NVCC__

#define SDIV(x,y)(((x)+(y)-1)/(y))

__forceinline__ __device__
realG3 warpReduceSumTriple(realG3 val, unsigned mask) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val.x += __shfl_down_sync(mask, val.x, offset);
    val.y += __shfl_down_sync(mask, val.y, offset);
    val.z += __shfl_down_sync(mask, val.z, offset);
  }
  return val; 
}
__forceinline__ __device__
realG warpReduceSum(realG val, unsigned mask) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down_sync(mask, val, offset);
  return val;
}
__forceinline__ __device__
realG blockReduceSum(realG val, unsigned mask) {

  static __shared__ int shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val, mask);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;

  if (wid==0) val = warpReduceSum(val, mask); //Final reduce within first warp

  return val;
}
__forceinline__ __device__
realG3 blockReduceSumTriple(realG3 val, unsigned mask) {

  static __shared__ realG3 shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSumTriple(val, mask);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : make_realG3(0.0,0.0,0.0,0.0);

  if (wid==0) val = warpReduceSumTriple(val, mask); //Final reduce within first warp

  return val;
}

using namespace std;
#define CUERR { \
  cudaError_t cudaerr; \
  if ((cudaerr = cudaGetLastError()) != cudaSuccess){ \
      printf("CUDA ERROR: \"%s\" in File %s at LINE %d.\n", cudaGetErrorString(cudaerr), __FILE__, __LINE__); \
  } \
}

#define PRINTL { \
  if(threadIdx.x == 0){ \
    printf("Line: %d\n", __LINE__); \
  } \
}
#endif
