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

    template <typename T=float>
    void check_dense (np::ndarray array, int nd = 0) {
        CHECK(array.get_dtype() == np::dtype::get_builtin<T>());
        if (nd > 0) {
            CHECK(array.get_nd() == nd);
        }
        else {
            nd = array.get_nd();
        }
        int stride = sizeof(T);
        for (int i = 0, off=nd-1; i < nd; ++i, --off) {
            CHECK(array.strides(off) == stride);
            stride *= array.shape(off);
        }
    }

    class Box {
        float x, y, z, l, w, h, t, d;
        cv::Rect_<float> rect;

        static cv::Rect_<float> make_rect (cv::Point2f const &pt,
                                    float width, float height) {
            return cv::Rect_<float>(pt.x - width/2,
                                    pt.y - height/2,
                                    width, height);
        }
    public:
        void load (float const *params) {
            x = params[0];
            y = params[1];
            z = params[2];
            h = params[3];
            w = params[4];
            l = params[5];
            t = params[6];
            d = params[7];
            rect = make_rect(cv::Point2f(x, y), d, d);
        }

        void store (float *params) {
            params[0] = x;
            params[1] = y;
            params[2] = z;
            params[3] = h;
            params[4] = w;
            params[5] = l;
            params[6] = t;
            params[7] = d;
        }

        py::tuple make_tuple () const {
            return py::make_tuple(x, y, z, h, w, l, t, d);
        }

        void to_residual (cv::Point2f pt, float const *prior, float *params) const {
            params[0] = x - pt.x;
            params[1] = y - pt.y;
            params[2] = z;
            params[3] = h;
            params[4] = w;
            params[5] = l;
            params[6] = t;
            params[7] = d;
        }

        void from_residual (cv::Point2f pt, float const *prior, float const *params) {
            x = params[0] + pt.x;
            y = params[1] + pt.y;
            z = params[2];
            h = params[3];
            w = params[4];
            l = params[5];
            t = params[6];
            d = params[7];
            rect = make_rect(cv::Point2f(x, y), d, d);
        }

        float score_anchor (cv::Point2f pt, float const *prior) const {
            cv::Rect_<float> p = make_rect(pt, prior[0], prior[1]);
            cv::Rect_<float> u = p & rect;
            float i = u.area();
            return i / (p.area() + rect.area() - i + 0.00001);
        }
    };

    class Boxes: public vector<Box> {
    public:
        Boxes (np::ndarray array) {
            check_dense<float>(array, 2);
            resize(array.shape(0));
            for (int b = 0; b < array.shape(0); ++b) {
                at(b).load((float const *)(array.get_data() + b * array.strides(0)));
            }
        };
    };


    class Voxelizer {
        float x_min, x_max, x_factor;
        float y_min, y_max, y_factor;
        float z_min, z_max, z_factor;
        int nx, ny, nz;
        float lower_th, upper_th;
        std::default_random_engine rng;
    public:
        Voxelizer (np::ndarray ranges,
                   np::ndarray shape, float lower_th_, float upper_th_): lower_th(lower_th_), upper_th(upper_th_) {
            // ranges is a 3 * 2 arrange
            {
                check_dense<float>(ranges, 2);
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
                for (int j = 0; j < N; ++j) index[j] = j;
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
                    if (int(voxels[cell].size()) < T) {
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

        py::tuple voxelize_labels (py::list list, np::ndarray priors, int downsize) {
            check_dense<float>(priors, 2);
            int batch = py::len(list);
            CHECK(nz % downsize == 0);
            CHECK(ny % downsize == 0);
            CHECK(nx % downsize == 0);
            int lx = nx / downsize;
            int ly = ny / downsize;
            //int lz = nz / downsize;

            CHECK(priors.get_nd() == 2);
            int params = 8; //priors.shape(1);
            // anchors and masks
            np::ndarray A = np::zeros(py::make_tuple(batch, lx, ly, priors.shape(0)), np::dtype::get_builtin<float>());
            check_dense<float>(A);
            np::ndarray AW = np::zeros(py::make_tuple(batch, lx, ly, priors.shape(0)), np::dtype::get_builtin<float>());
            AW += 1.0;
            check_dense<float>(AW);
            // parameters and masks
            np::ndarray P = np::zeros(py::make_tuple(batch, lx, ly, priors.shape(0) * params), np::dtype::get_builtin<float>());
            check_dense<float>(P);
            np::ndarray PW = np::zeros(py::make_tuple(batch, lx, ly, priors.shape(0)), np::dtype::get_builtin<float>());
            check_dense<float>(PW);

            int count = 0;
            for (int i = 0; i < batch; ++i) {
                Boxes boxes(py::extract<np::ndarray>(list[i]));
                if (boxes.empty()) continue;
                float min_z = 100;
                float max_z = -100;
                
                /*
                for (int b = 0; b < boxes.shape(0); ++b) {
                    float const *box = (float const *)(boxes.get_data() + b * boxes.strides(0));
                    if (box[2] > max_z) max_z = box[2];
                    if (box[2] < min_z) min_z = box[2];
                    std::cerr << "BOX " << box[0] << ' ' << box[1] << ' ' << box[4] << ' ' << box[5] << ' ' << box[7] << std::endl;
                }
                std::cerr << "Z " << min_z << " " << max_z << std::endl;
                */

                float *pa = (float *)(A.get_data() + A.strides(0) * i);
                float *paw = (float *)(AW.get_data() + AW.strides(0) * i);
                float *pp = (float *)(P.get_data() + P.strides(0) * i);
                float *ppw = (float *)(PW.get_data() + PW.strides(0) * i);
                for (int x = 0; x < lx; ++x) {
                    for (int y = 0; y < ly; ++y) {
                        // find closest shape
                        cv::Point2f pt(x * downsize / x_factor + x_min, y * downsize / y_factor + y_min);
                        for (int k = 0; k < priors.shape(0); ++k,
                                pa += 1, paw +=1,
                                pp += params, ppw += 1) {
                            float const *prior = (float const *)(priors.get_data() + k * priors.strides(0));
                            Box const *best_box = nullptr;
                            float best_d = 0;
                            for (auto const &box: boxes) {
                                float d = box.score_anchor(pt, prior); 
                                if (d > best_d) {   // find best circle
                                    best_d = d;
                                    best_box = &box;
                                }
                            }
                            if (!best_box) continue;
                            if (best_d >= lower_th) {
                                best_box->to_residual(pt, prior, pp);
                                ppw[0] = 1.0; //best_c->weight;
                                ++count;
                                if (best_d < upper_th) {
                                    paw[0] = 0;
                                }
                                else {
                                    pa[0] = 1;      // to class label
                                }
                            }
                        } // prior
                    } // x
                } // y
            } // batch
            return py::make_tuple(A, AW, P, PW);
        }

        py::list generate_boxes (np::ndarray probs, np::ndarray params, np::ndarray priors, float anchor_th) {
            check_dense<float>(probs, 4);
            check_dense<float>(params, 4);
            check_dense<float>(priors, 2);
            int batch = probs.shape(0);
            int lx = probs.shape(1); //nx / downsize;
            int ly = probs.shape(2); //ny / downsize;
            CHECK(nx % lx == 0);
            CHECK(ny % ly == 0);
            int downsize = nx / lx;
            CHECK(ny / ly == downsize);
            CHECK(params.shape(1) == lx);
            CHECK(params.shape(2) == ly);
            CHECK(params.shape(3) % probs.shape(3) == 0);
            CHECK(probs.shape(3) == priors.shape(0));
            int n_params = params.shape(3) / probs.shape(3);

            py::list list;
            for (int i = 0; i < batch; ++i) {
                float *pa = (float *)(probs.get_data() + probs.strides(0) * i);
                float *pp = (float *)(params.get_data() + params.strides(0) * i);
                py::list boxes;
                for (int x = 0; x < lx; ++x) {
                    for (int y = 0; y < ly; ++y) {
                        cv::Point2f pt(x * downsize / x_factor + x_min, y * downsize / y_factor + y_min);
                        // check all the priors
                        for (int k = 0; k < probs.shape(3); ++k, pa += 1, pp += n_params) {
                            float prob = pa[0];
                            if (prob < anchor_th) continue;
                            float const *prior = (float const *)(priors.get_data() + k * priors.strides(0));
                            Box box;
                            box.from_residual(pt, prior, pp);
                            boxes.append(box.make_tuple());
                        } // prior
                    } // x
                } // y
                list.append(boxes);
            } // batch
            return list;
        }

        void augment (np::ndarray points, np::ndarray boxes) {
            // rotate by z
            // shift by z
            std::uniform_real_distribution<float> scale(0.95, 1.05);
            std::uniform_real_distribution<float> rotate(-M_PI/4, M_PI/4);
            float s = scale(rng);
            float r = rotate(rng);
            // TODO
        }
    };
}

BOOST_PYTHON_MODULE(cpp)
{
    np::initialize();
    py::class_<Voxelizer>("Voxelizer", py::init<np::ndarray, np::ndarray, float, float>())
        .def("voxelize_points", &Voxelizer::voxelize_points)
        .def("make_dense", &Voxelizer::make_dense)
        .def("voxelize_labels", &Voxelizer::voxelize_labels)
        .def("generate_boxes", &Voxelizer::generate_boxes)
        .def("augment", &Voxelizer::augment)
    ;
}

