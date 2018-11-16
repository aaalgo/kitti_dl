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
        float x_min, x_max, x_factor;
        float y_min, y_max, y_factor;
        float z_min, z_max, z_factor;
        int nx, ny, nz;
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
                x_min = *ptr++;
                x_max = *ptr++;
                y_min = *ptr++;
                y_max = *ptr++;
                z_min = *ptr++;
                z_max = *ptr++;
            }
            {
                CHECK(shape.get_nd() == 1);
                CHECK(shape.get_dtype() == np::dtype::get_builtin<int>());
                int D = shape.shape(0);
                CHECK(D == 3);
                int const *ptr = (int const *)shape.get_data();
                nx = *ptr++;
                ny = *ptr++;
                nz = *ptr++;
            }

            float x_range = x_max - x_min;
            float y_range = y_max - y_min;
            float z_range = z_max - z_min;

            // quantization method
            // [min, max], range   -->   [0, n-1]
            // q(x) = round(((x - min)/range) * (n-1))
            //      = round((x - min) * factor)
            x_factor = (nx - 1) / x_range;
            y_factor = (ny - 1) / y_range;
            z_factor = (nz - 1) / z_range;
        }

        static int quantize (float v, float min, float factor) {
            return int(round((v-min)*factor));
        }

        // group point by voxels
        py::tuple voxelize_points (py::list list, int T) { // np::ndarray points, int T) {
            int batch = py::len(list);
            CHECK(batch > 0);
            int Cin;
            {   // checkout the first item in batch
                np::ndarray points = py::extract<np::ndarray>(list[0]);
                CHECK(points.get_nd() == 2);
                CHECK(points.get_dtype() == np::dtype::get_builtin<float>());
                Cin = points.shape(1);
                CHECK(Cin == 4);
            }
            int C = Cin + 3;
            int nv = nx * ny * nz;

            vector<vector<float const*>> voxels(batch * nv);
            for (int i = 0; i < batch; ++i) {
                // foreach batch
                np::ndarray points = py::extract<np::ndarray>(list[i]);
                CHECK(points.shape(1) == Cin);
                CHECK(points.get_nd() == 2);
                CHECK(points.get_dtype() == np::dtype::get_builtin<float>());
                int N = points.shape(0);
                CHECK(points.strides(0) == Cin * int(sizeof(float)));
                CHECK(points.strides(1) == int(sizeof(float)));

                vector<unsigned> index(N);
                for (unsigned j = 0; j < N; ++j) index[j] = j;
                std::random_shuffle(index.begin(), index.end());

                for (unsigned j: index) {
                    float const *ptr = (float const *)(points.get_data() + points.strides(0) * j);
                    float x = ptr[0];
                    float y = ptr[1];
                    float z = ptr[2];

                    int ix = quantize(x, x_min, x_factor);
                    if (ix < 0 || ix >= nx) continue;
                    int iy = quantize(y, y_min, y_factor);
                    if (iy < 0 || iy >= ny) continue;
                    int iz = quantize(z, z_min, z_factor);
                    if (iz < 0 || iz >= nz) continue;

                    int cell = i * nv + (ix * ny + iy) * nz + iz;
                    if (voxels[cell].size() < T) {
                        voxels[cell].push_back(ptr);
                    }
                }
            }

            int NZ = 0; // non-empty entries
            for (auto const &v: voxels) {
                if (v.size() > 0) ++NZ;
            }
            // voxel points
            np::ndarray P = np::zeros(py::make_tuple(NZ, T, C), np::dtype::get_builtin<float>());
            int P_voxel_stride = T * C;
            CHECK(P_voxel_stride * int(sizeof(float)) == P.strides(0));
            float *pp = (float *)P.get_data();

            // voxel lengths (# points in voxel)
            np::ndarray M = np::zeros(py::make_tuple(NZ, T, 1), np::dtype::get_builtin<float>());
            int M_voxel_stride = T;
            CHECK(T * int(sizeof(float)) == M.strides(0));
            float *mm = (float *)M.get_data();

            np::ndarray I = np::zeros(py::make_tuple(NZ), np::dtype::get_builtin<int32_t>());
            int32_t *ii = (int32_t *)I.get_data();

            for (unsigned j = 0; j < voxels.size(); ++j) {
                auto const &voxel = voxels[j];
                if (voxel.empty()) continue;
                *ii++ = j;

                float *p = pp; // copy points within voxel
                float *m = mm;
                float x = 0, y = 0, z = 0;
                for (float const *from: voxel) {
                    x += from[0];
                    y += from[1];
                    z += from[2];
                }
                float cnt = voxel.size();
                if (cnt > 0) {
                    x /= cnt;
                    y /= cnt;
                    z /= cnt;
                }
                for (float const *from: voxel) {
                    p[0] = from[0];
                    p[1] = from[1];
                    p[2] = from[2];
                    p[3] = from[3];
                    p[4] = from[0]-x;
                    p[5] = from[1]-y;
                    p[6] = from[2]-z;
                    p += C;
                    *m++ = 1.0;
                }
                pp += P_voxel_stride;
                mm += M_voxel_stride;
            }
            return py::make_tuple(P, M, I);
        }

        py::tuple make_dense (np::ndarray P, np::ndarray I) {
            CHECK(P.get_nd() == 2);
            CHECK(P.get_dtype() == np::dtype::get_builtin<float>());
            int C = P.shape(1);
            int nv = nx * ny * nz;
            CHECK(I.get_nd() == 1);
            CHECK(P.shape(0) == I.shape(0));
            int N = P.shape(0);

            // infer batch from last index in I
            int batch = 0;
            {
                int32_t const *ii = (int32_t const *)(I.get_data() + (N-1) * I.strides(0));
                batch = (ii[0] / nv) + 1;
            }

            np::ndarray V = np::zeros(py::make_tuple(batch, nx, ny, nz, C), np::dtype::get_builtin<float>());
            CHECK(nv * C * int(sizeof(float)) == V.strides(0));
            float *vv = (float *)V.get_data();

            for (int i = 0; i < N; ++i) {
                float const *pp = (float const *)(P.get_data() + i * P.strides(0));
                int32_t const *ii = (int32_t const *)(I.get_data() + i * I.strides(0));
                float *oo = vv + ii[0] * C;
                std::copy(pp, pp + C, oo);
            }
            return py::make_tuple(V);
        }

        py::tuple voxelize_labels (py::list list, np::ndarray priors, int downsize) { //np::ndarray labels, int downsize) {
            int batch = py::len(list);
            CHECK(nz % downsize == 0);
            CHECK(ny % downsize == 0);
            CHECK(nx % downsize == 0);
            int lx = nx / downsize;
            int ly = ny / downsize;
            //int lz = nz / downsize;

            CHECK(priors.get_nd() == 2);
            int params = 4; //priors.shape(1);
            // anchors and masks
            np::ndarray A = np::zeros(py::make_tuple(batch, lx, ly, priors.shape(0)), np::dtype::get_builtin<float>());
            np::ndarray AW = np::zeros(py::make_tuple(batch, lx, ly, priors.shape(0)), np::dtype::get_builtin<float>());
            // parameters and masks
            np::ndarray P = np::zeros(py::make_tuple(batch, lx, ly, priors.shape(0) * params), np::dtype::get_builtin<float>());
            np::ndarray PW = np::zeros(py::make_tuple(batch, lx, ly, priors.shape(0)), np::dtype::get_builtin<float>());

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
        .def("make_dense", &Voxelizer::make_dense)
        .def("voxelize_labels", &Voxelizer::voxelize_labels)
    ;
}

