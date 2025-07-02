#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "voxelizer.h"

namespace bind = pybind11;

PYBIND11_MODULE(voxelizer, m) {
  m.def("voxelize", &voxelize,
        "GPU-accelerated voxelization",
        bind::arg("points"),
        bind::arg("voxel_size"),
        bind::arg("coors_range"),
        bind::arg("max_points_per_voxel") = 32,
        bind::arg("max_voxels") = 20000
  );
}
