__global__ void fill_voxels(
    const float* __restrict__ pts,      // [N×(3+F)] 
    const int32_t* __restrict__ p2v,    // [N] → voxel‐index or -1
    float*       __restrict__ voxels,   // [V, K, 3+F]
    int32_t*     __restrict__ counts,   // [V]
    int           N, int V, int K, int D  // D = 3(+features)
) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= N) return;
  int v = p2v[i];
  if (v < 0 || v >= V) return;
  // reserve a slot via atomic:
  int slot = atomicAdd(&counts[v], 1);
  if (slot < K) {
    // copy D floats from pts + i*D into voxels[v,K,i]
    float* dst = voxels + (v*K + slot)*D;
    const float* src = pts   + i*D;
    #pragma unroll
    for (int d = 0; d < D; ++d) dst[d] = src[d];
  }
}
