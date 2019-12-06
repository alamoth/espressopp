#define realG double
#define realG3 double3
#define realG4 double3
#define make_realG3 make_double3
#define make_realG4 make_double3
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
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : make_realG3(0.0,0.0,0.0);

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
#ifndef CUTIL_MATH_H
#define CUTIL_MATH_H

inline __host__ __device__ double3 operator+(double3 a, double3 b)
{
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(double3 &a, double3 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}
inline __host__ __device__ double3 operator+(double3 a, double b)
{
    return make_double3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(double3 &a, double b)
{
    a.x += b; a.y += b; a.z += b;
}

inline __host__ __device__ double4 operator+(double4 a, double4 b)
{
    return make_double4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ void operator+=(double4 &a, double4 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}
inline __host__ __device__ double4 operator+(double4 a, double b)
{
    return make_double4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ double4 operator+(double b, double4 a)
{
    return make_double4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ void operator+=(double4 &a, double b)
{
    a.x += b; a.y += b; a.z += b; a.w += b;
}

inline __host__ __device__ double4 operator-(double4 a, double4 b)
{
    return make_double4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ void operator-=(double4 &a, double4 b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}
inline __host__ __device__ double4 operator-(double4 a, double b)
{
    return make_double4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline __host__ __device__ void operator-=(double4 &a, double b)
{
    a.x -= b; a.y -= b; a.z -= b; a.w -= b;
}

inline __host__ __device__ double3 operator-(double3 a, double3 b)
{
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(double3 &a, double3 b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
inline __host__ __device__ double3 operator-(double3 a, double b)
{
    return make_double3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ double3 operator-(double b, double3 a)
{
    return make_double3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(double3 &a, double b)
{
    a.x -= b; a.y -= b; a.z -= b;
}

inline __host__ __device__ double3 operator*(double3 a, double3 b)
{
    return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(double3 &a, double3 b)
{
    a.x *= b.x; a.y *= b.y; a.z *= b.z;
}
inline __host__ __device__ double3 operator*(double3 a, double b)
{
    return make_double3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ double3 operator*(double b, double3 a)
{
    return make_double3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(double3 &a, double b)
{
    a.x *= b; a.y *= b; a.z *= b;
}

inline __host__ __device__ double4 operator*(double4 a, double4 b)
{
    return make_double4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline __host__ __device__ void operator*=(double4 &a, double4 b)
{
    a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w;
}
inline __host__ __device__ double4 operator*(double4 a, double b)
{
    return make_double4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline __host__ __device__ double4 operator*(double b, double4 a)
{
    return make_double4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(double4 &a, double b)
{
    a.x *= b; a.y *= b; a.z *= b; a.w *= b;
}

inline __host__ __device__ double dot(double3 a, double3 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ double dot(double4 a, double4 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
#endif