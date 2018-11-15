#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <boost/ref.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
namespace py = boost::python;
namespace np = boost::python::numpy;

namespace {
    using std::istringstream;
    using std::ostringstream;
    using std::string;
    using std::runtime_error;
    using std::cerr;
    using std::endl;
    using std::vector;

    class Voxelizer {
        float z_min, z_max, z_factor;
        float y_min, y_max, y_factor;
        float x_min, x_max, x_factor;
        int nz, ny, nx;
    public:
        Voxelizer (np::ndarray ranges,
                   np::ndarray shape) {
            // ranges is a 3 * 2 arrange
            {
                CHECK(ranges.get_nd() == 2);
                CHECK(ranges.get_dtype() == np::dtype::get_builtin<float>());
                int D = ranges.shape(0);
                int W = ranges.shape(1);
                CHECK(D == 3);
                CHECK(W == 2);
                float const *ptr = (float const *)ranges.get_data();
                z_min = *ptr++;
                z_max = *ptr++;
                y_min = *ptr++;
                y_max = *ptr++;
                x_min = *ptr++;
                x_max = *ptr++;
            }
            {
                CHECK(shape.get_nd() == 1);
                CHECK(shape.get_dtype() == np::dtype::get_builtin<int>());
                int D = shape.shape(0);
                CHECK(D == 3);
                int const *ptr = (int const *)shape.get_data();
                nz = *ptr++;
                ny = *ptr++;
                nx = *ptr++;
            }

            float z_range = z_max - z_min;
            float y_range = y_max - y_min;
            float x_range = x_max - x_min;

            // quantization method
            // [min, max], range   -->   [0, n-1]
            // q(x) = round(((x - min)/range) * (n-1))
            //      = round((x - min) * factor)
            z_factor = (nz - 1) / z_range;
            y_factor = (ny - 1) / y_range;
            x_factor = (nx - 1) / x_range;
        }

        static int quantize (float v, float min, float factor) {
            return int(round((v-min)*factor));
        }

        // group point by voxels
        py::tuple voxelize_points (py::list list, int T) { // np::ndarray points, int T) {
            int batch = py::len(list);
            CHECK(batch > 0);
            int C;
            {   // checkout the first item in batch
                np::ndarray points = py::extract<np::ndarray>(list[0]);
                CHECK(points.get_nd() == 2);
                CHECK(points.get_dtype() == np::dtype::get_builtin<float>());
                C = points.shape(1);
            }
            //py::tuple ret;
            int nv = nx * ny * nz;
            // voxel points
            np::ndarray P = np::zeros(py::make_tuple(batch, nv, T, C), np::dtype::get_builtin<float>());
            int P_voxel_stride = T * C;
            CHECK(nv * P_voxel_stride * int(sizeof(float)) == P.strides(0));
            float *pp = (float *)P.get_data();

            // voxel lengths (# points in voxel)
            np::ndarray L = np::zeros(py::make_tuple(batch, nv, 1), np::dtype::get_builtin<int>());
            CHECK(nv * int(sizeof(int)) == L.strides(0));
            float *ll = (float *)L.get_data();

            vector<vector<float const*>> voxels(nv);
            for (int i = 0; i < batch; ++i) {//, pp += P_batch_stride, ll += L_batch_stride) {
                // foreach batch
                np::ndarray points = py::extract<np::ndarray>(list[i]);
                CHECK(points.shape(1) == C);
                CHECK(points.get_nd() == 2);
                CHECK(points.get_dtype() == np::dtype::get_builtin<float>());
                int N = points.shape(0);
                CHECK(points.strides(0) == C * int(sizeof(float)));
                CHECK(points.strides(1) == int(sizeof(float)));

                for (auto &v: voxels) v.clear();
                // distribute points to voxels
                for (int j = 0; j < N; ++j) {
                    // one point
                    float const *ptr = (float const *)(points.get_data() + points.strides(0) * j);
                    // TODO: check order of xyz
                    float x = ptr[0];
                    float y = ptr[1];
                    float z = ptr[2];

                    int ix = quantize(x, x_min, x_factor);
                    if (ix < 0 || ix >= nx) continue;
                    int iy = quantize(y, y_min, y_factor);
                    if (iy < 0 || iy >= ny) continue;
                    int iz = quantize(z, z_min, z_factor);
                    if (iz < 0 || iz >= nz) continue;

                    int cell = (iz * ny + iy) * nx + ix;
                    voxels[cell].push_back(ptr);
                }

                // copy the voxels to output
                for (auto &voxel: voxels) {
                    if (voxel.size() > unsigned(T)) { 
                        std::random_shuffle(voxel.begin(), voxel.end());
                        voxel.resize(T);
                    }
                    *ll++ = voxel.size();
                    float *p = pp; // copy points within voxel
                    for (float const *from: voxel) {
                        std::copy(from, from + C, p);
                        p += C;
                    }
                    pp += P_voxel_stride;
                }
            }
            return py::make_tuple(P, L);
        }

        py::tuple voxelize_labels (py::list list, np::ndarray priors, int downsize) { //np::ndarray labels, int downsize) {
            int batch = py::len(list);
            CHECK(nz % downsize == 0);
            CHECK(ny % downsize == 0);
            CHECK(nx % downsize == 0);
            int lz = nz / downsize;
            int ly = ny / downsize;
            int lx = nx / downsize;

            CHECK(priors.get_nd() == 2);
            int params = priors.shape(1);
            // anchors and masks
            np::ndarray A = np::zeros(py::make_tuple(batch, lz, ly, lx, priors.shape(0)), np::dtype::get_builtin<float>());
            np::ndarray AW = np::zeros(py::make_tuple(batch, lz, ly, lx, priors.shape(0)), np::dtype::get_builtin<float>());
            // parameters and masks
            np::ndarray P = np::zeros(py::make_tuple(batch, lz, ly, lx, priors.shape(0), params), np::dtype::get_builtin<float>());
            np::ndarray PW = np::zeros(py::make_tuple(batch, lz, ly, lx, priors.shape(0)), np::dtype::get_builtin<float>());

            /*
            int P_voxel_stride = T * C;
            CHECK(nv * P_voxel_stride * int(sizeof(float)) == P.strides(0));
            float *pp = (float *)P.get_data();
            */
            return py::make_tuple(A, AW, P, PW);
        }
    };
}

BOOST_PYTHON_MODULE(cpp)
{
    np::initialize();
    py::class_<Voxelizer>("Voxelizer", py::init<np::ndarray, np::ndarray>())
        .def("voxelize_points", &Voxelizer::voxelize_points)
        .def("voxelize_labels", &Voxelizer::voxelize_labels)
    ;
}

