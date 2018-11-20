#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <boost/ref.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
//#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <hdf5_hl.h>
#include "streamer/streamer.h"
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

    static int PARAMS = 8;
    int random_seed = 2019;

    // sanity check np::ndarray to make sure they are dense and continuous
    template <typename T=float>
    void check_dense (np::ndarray array, int nd = 0) {
        CHECK(array.get_dtype() == np::dtype::get_builtin<T>());
        if (nd > 0) CHECK(array.get_nd() == nd);
        else nd = array.get_nd();
        int stride = sizeof(T);
        for (int i = 0, off=nd-1; i < nd; ++i, --off) {
            CHECK(array.strides(off) == stride);
            stride *= array.shape(off);
        }
    }

    // allocate an ndarray of given shape, return the pointer to firt element
    template <typename T=float>
    T *alloc_ndarray (py::tuple shape, np::ndarray **ptr) {
        np::ndarray *array = new np::ndarray(np::zeros(shape, np::dtype::get_builtin<T>()));
        check_dense<T>(*array);
        *ptr = array;
        return reinterpret_cast<T *>(array->get_data());
    }

    // Lidar point, xyz and reflectance
    struct __attribute__((__packed__)) Point {
        static int constexpr DIM = 4;
        union {
            float data[4];
            struct {
                float x, y, z, r;
            };
        };
    };

    struct __attribute__((__packed__)) Prior {
        static int constexpr DIM = 2;
        union {
            float data[2];
            struct {
                float l, w; //, h, t;   // length is along x, width is along y
            };
        };
    };

    // 3D rotated box
    struct __attribute__((__packed__)) Box {
        static int constexpr DIM = 8;
        union {
            float data[8];
            struct {
                float x, y, z, h, w, l, t, s;
            };
        };
    public:
        void load (float const *params) {
            std::copy(params, params + DIM, data);
        }

        void store (float *params) {
            std::copy(data, data + DIM, params);
        }

        py::tuple make_tuple () const {
            return py::make_tuple(x, y, z, h, w, l, t, s);
        }

        void to_residual (float ax, float ay, Prior const &, float *params) const {
            // residual is the regression target
            params[0] = x - ax;
            params[1] = y - ay;
            params[2] = z;
            params[3] = h; params[4] = w; params[5] = l; params[6] = t; params[7] = 0;
        }

        void from_residual (float ax, float ay, Prior const &, float const *params) {
            x = params[0] + ax;
            y = params[1] + ay;
            z = params[2];
            h = params[3]; w = params[4]; l = params[5]; t = params[6];
        }

        static float intersect (float min1, float max1, float min2, float max2) {
            // intersection length of range [min1, max1] and [min2, max2]
            float a = std::max(min1, min2);
            float b = std::min(max1, max2);
            if (a <= b) return b-a;
            return 0;
        }

        float score_anchor (float ax, float ay, Prior const &prior) const {
            float a = x * x + y * y;
            float d = sqrt(a)/2;
            float p = prior.l * prior.w;
            float i = intersect(x-d, x+d, ax-prior.l/2, ax+prior.l/2)   // intersection area
                     * intersect(y-d, y+d, ay-prior.w/2, ay+prior.w/2);
            return i / (a + p - i + 0.00001);
        }
    };

    // Give unmanaged memory an array-like accessment interface.
    template <typename T>
    class View {
        T const *data;
        size_t sz;
    public:
        View (void *p, size_t s): data(reinterpret_cast<T *>(p)), sz(s) {
        }

        View (np::ndarray array) {
			check_dense<float>(array, 2);
			CHECK(array.shape(1) * sizeof(float) == sizeof(T));
			data = reinterpret_cast<T *>(array.get_data());
			sz = array.shape(0);
        }

        View (vector<T> const &v) {
            data = &v[0];
            sz = v.size();
        }

        T const &operator [] (size_t i) const {
            return data[i];
        }
        size_t size () const { return sz; }

        bool empty () const { return sz == 0; }

        T const *begin () const { return data; }
        T const *end () const { return data + sz; }
    };

    class H5File {
        hid_t hid;
    public:
        H5File (string const &path) {
            hid = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            CHECK(hid >= 0);
        }

        template <typename T>
        void load (char const *name, vector<T> *buffer) {
            herr_t e = H5LTfind_dataset(hid, name);
            int rank;
            e = H5LTget_dataset_ndims(hid, name, &rank);
            CHECK(e >= 0);
            CHECK(rank == 2);
            vector<hsize_t> dims(rank);
            H5T_class_t class_id;
            size_t type_size;
            e = H5LTget_dataset_info(hid, name, &dims[0], &class_id, &type_size);
            CHECK(class_id == H5T_FLOAT);
            CHECK(type_size == sizeof(float));
            CHECK(dims[1] * sizeof(float) == sizeof(T));
            buffer->resize(dims[0]);
            if (buffer->size() > 0) {
                e = H5LTread_dataset_float(hid, name, buffer->at(0).data);
            }
            CHECK(e >= 0);
        }

        ~H5File () {
            herr_t e = H5Fclose(hid);
            CHECK(e >= 0);
        }
    };

    class Voxelizer {
        float x_min, x_max, x_factor;
        float y_min, y_max, y_factor;
        float z_min, z_max, z_factor;
        int nx, ny, nz;
        std::default_random_engine rng;
    public:
        Voxelizer (np::ndarray ranges, np::ndarray shape): rng(random_seed) {
            // ranges is a 3 * 2 arrange
            {
                check_dense<float>(ranges, 2);
                CHECK(ranges.shape(0) == 3);
                CHECK(ranges.shape(1) == 2);
                float const *ptr = (float const *)ranges.get_data();
                x_min = *ptr++; x_max = *ptr++;
                y_min = *ptr++; y_max = *ptr++;
                z_min = *ptr++; z_max = *ptr++;
            }
            {
                check_dense<int>(shape, 1);
                CHECK(shape.shape(0) == 3);
                int32_t const *ptr = (int32_t const *)shape.get_data();
                nx = *ptr++; ny = *ptr++; nz = *ptr++;
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

        // group point by voxels in the grid nx * ny * nz
        // This is the helper function that do the real work.
        void voxelize_points_helper (vector<View<Point>> const &points_batch, int T, bool lock,
                // the return arrays, P for points, M for masks and I for voxel indexes
                np::ndarray **V, np::ndarray **M, np::ndarray **I) {
            // Input:
            //  points_batch:    a batch of point clouds; each as View<Point>.
            //  T:      maximal points in each voxel
            //  lock:   when allocating ndarray, should python thread be locked
            // Output:
            //  *V:     pointer to voxel array,  N * T * C
            //          N non-empty voxels, each voxel has T points, each point has C channels
            //  *M:     pointer to voxel mask,   N * T
            //          For each non empty voxel, T 0/1 values.  1 for those with valid points.
            //  *I:     flatten voxel index,     N
            //          We use this index to put the voxels back into a dense 3D grid

            int batch = points_batch.size();
            int Cin = 4;
            int C = Cin + 3;
            int nv = nx * ny * nz;

            vector<vector<Point const *>> voxels(batch * nv);
            for (int i = 0; i < batch; ++i) {
                // foreach batch
                auto const &points = points_batch[i];

                // we need to drop points from over-crowded voxels in randomm
                // and we do this by shuffling the points before adding them to voxels
                vector<unsigned> index(points.size());
                for (unsigned j = 0; j < index.size(); ++j) index[j] = j;
                std::random_shuffle(index.begin(), index.end());

                for (unsigned j: index) {
                    Point const &point = points[j];

                    // TODO:
                    //      some points may be around the voxel boundary
                    //      we might want to added in those close to boundary as well
                    int ix = quantize(point.x, x_min, x_factor);
                    if (ix < 0 || ix >= nx) continue;
                    int iy = quantize(point.y, y_min, y_factor);
                    if (iy < 0 || iy >= ny) continue;
                    int iz = quantize(point.z, z_min, z_factor);
                    if (iz < 0 || iz >= nz) continue;

                    int cell = i * nv + (ix * ny + iy) * nz + iz;
                    auto &voxel = voxels[cell];
                    if (int(voxel.size()) < T) {
                        voxel.push_back(&point);
                    }
                }
            }

            int N = 0; // non-empty entries
            for (auto const &v: voxels) if (v.size() > 0) ++N;

            float *V_buf, *M_buf;
            int32_t *I_buf;

            {   // allocate ndarrays for return values
                streamer::ScopedGState _(lock);
                V_buf = alloc_ndarray<>(py::make_tuple(N, T, C), V);
                M_buf = alloc_ndarray<>(py::make_tuple(N, T, 1), M);
                I_buf = alloc_ndarray<int32_t>(py::make_tuple(N), I);
            }
            // fill points into voxels
            for (unsigned j = 0; j < voxels.size(); ++j) {
                auto const &voxel = voxels[j];
                if (voxel.empty()) continue;
                *I_buf++ = j;

                float *p = V_buf, *m = M_buf;
                V_buf += T * C; M_buf += T;

                float cx = 0, cy = 0, cz = 0;   // centroid of points in voxel
                for (Point const *point: voxel) {
                    cx += point->x;
                    cy += point->y;
                    cz += point->z;
                }
                cx /= voxel.size();     // we only handle non-empty voxels here
                cy /= voxel.size();     // so we can directly divide
                cz /= voxel.size();
                for (Point const *point: voxel) {
                    *p++ = point->x;    *p++ = point->y;    *p++ = point->z;
                    *p++ = point->r;
                    *p++ = point->x-cx;   *p++ = point->y-cy;   *p++ = point->z-cz;
                    *m++ = 1.0;
                }
            }
        }

        // group point by voxels
        py::tuple voxelize_points (py::list points_batch, int T) {
            int batch = py::len(points_batch);
            CHECK(batch > 0);
            vector<View<Point>> views;
            for (int i = 0; i < batch; ++i) {
                np::ndarray points = py::extract<np::ndarray>(points_batch[i]);
                check_dense<float>(points, 2);
                CHECK(points.shape(1) == 4);
                views.emplace_back(points);
            }
            np::ndarray *V, *M, *I;
            voxelize_points_helper(views, T, false, &V, &M, &I);
            py::tuple tuple = py::make_tuple(*V, *M, *I);
            delete V; delete M; delete I;
            return tuple;
        }

        // Generate one batch of label arrays
        // This is the helper function that do the real work.
        void voxelize_labels_helper (vector<View<Box>> const &boxes_batch, View<Prior> priors,
                int downsize, float lower_th, float upper_th, bool lock,
                np::ndarray **A, np::ndarray **AW, np::ndarray **P, np::ndarray **PW) {

            int batch = boxes_batch.size();

            CHECK(nz % downsize == 0);
            CHECK(ny % downsize == 0);
            CHECK(nx % downsize == 0);
            int lx = nx / downsize;
            int ly = ny / downsize;

            // anchors and masks
            float *pa, *paw, *pp, *ppw;
            {   // allocate returned ndarrays
                streamer::ScopedGState _(lock);
                pa = alloc_ndarray<>(py::make_tuple(batch, lx, ly, priors.size()), A);
                paw = alloc_ndarray<>(py::make_tuple(batch, lx, ly, priors.size()), AW);
                **AW += 1.0;
                pp = alloc_ndarray(py::make_tuple(batch, lx, ly, priors.size() * PARAMS), P);
                ppw = alloc_ndarray(py::make_tuple(batch, lx, ly, priors.size()), PW);
            }

            int count = 0;
            for (auto const &boxes: boxes_batch) {
                for (int x = 0; x < lx; ++x) { for (int y = 0; y < ly; ++y) {
                    float ax(x*downsize/x_factor+x_min);    // anchor location
                    float ay(y*downsize/y_factor+y_min);
                    for (auto const &prior: priors) {   // for each prior
                        Box const *best_box = nullptr;
                        float best_d = 0;
                        for (auto const &box: boxes) {  // for each box
                            float d = box.score_anchor(ax, ay, prior); 
                            if (d > best_d) {   // find best circle
                                best_d = d;
                                best_box = &box;
                            }
                        }
                        if (best_box && best_d >= lower_th) {
                            best_box->to_residual(ax, ay, prior, pp);
                            ppw[0] = 1.0; //best_c->weight;
                            ++count;
                            if (best_d < upper_th) {
                                paw[0] = 0;
                            }
                            else {
                                pa[0] = 1;      // to class label
                            }
                        }
                        pa += 1, paw +=1, pp += PARAMS, ppw += 1;
                    } // prior
                }} // y, x
            } // batch
        }

        py::tuple voxelize_labels (py::list boxes_batch, np::ndarray priors, int downsize, float lower_th, float upper_th) {
            vector<View<Box>> views;
            int batch = py::len(boxes_batch);
            CHECK(batch > 0);
            for (int i = 0; i < batch; ++i) {
                // foreach batch
                np::ndarray boxes = py::extract<np::ndarray>(boxes_batch[i]);
                check_dense<float>(boxes, 2);
                CHECK(boxes.shape(1) == 8);
                views.emplace_back(boxes);
            }
            View<Prior> priors_view(priors);
            np::ndarray *A, *AW, *P, *PW;
            voxelize_labels_helper(views, priors_view, downsize, lower_th, upper_th, false, &A, &AW, &P, &PW);
            py::tuple tuple = make_tuple(*A, *AW, *P, *PW);
            delete A; delete AW; delete P; delete PW;
            return tuple;
        }

        // put voxel feature to a dense grid of nx * ny * nz
        // return value is a tuple of one ndarray; tuple is needed to meet TF API
        py::tuple make_dense (np::ndarray P, np::ndarray I) {
            check_dense<float>(P, 2);
            int C = P.shape(1);
            int nv = nx * ny * nz;
            check_dense<int32_t>(I);
            int N = P.shape(0);
            CHECK(I.shape(0) == N);

            // infer batch from last index in I
            int batch = 0;
            {
                int32_t const *ii = (int32_t const *)(I.get_data() + (N-1) * I.strides(0));
                batch = (ii[0] / nv) + 1;
            }

            np::ndarray V = np::zeros(py::make_tuple(batch, nx, ny, nz, C), np::dtype::get_builtin<float>());
            check_dense<float>(V, 5);
            float *vv = (float *)V.get_data();

            for (int i = 0; i < N; ++i) {
                float const *pp = (float const *)(P.get_data() + i * P.strides(0));
                int32_t const *ii = (int32_t const *)(I.get_data() + i * I.strides(0));
                float *oo = vv + ii[0] * C;
                std::copy(pp, pp + C, oo);
            }
            return py::make_tuple(V);
        }

        py::list generate_boxes (np::ndarray probs, np::ndarray params, np::ndarray priors, float anchor_th) {
            check_dense<float>(probs, 4);
            check_dense<float>(params, 4);
            check_dense<float>(priors, 2);
            int batch = probs.shape(0);
            int lx = probs.shape(1);
            int ly = probs.shape(2);
            CHECK(nx % lx == 0);
            CHECK(ny % ly == 0);
            int downsize = nx / lx;
            CHECK(ny / ly == downsize);
            CHECK(params.shape(1) == lx);
            CHECK(params.shape(2) == ly);
            CHECK(params.shape(3) % probs.shape(3) == 0);
            CHECK(probs.shape(3) == priors.shape(0));
            CHECK(params.shape(3) / probs.shape(3) == PARAMS);

            View<Prior> priors_view(priors);

            py::list list;
            float *pa = (float *)(probs.get_data());
            float *pp = (float *)(params.get_data());
            for (int i = 0; i < batch; ++i) {
                py::list boxes;
                for (int x = 0; x < lx; ++x) { for (int y = 0; y < ly; ++y) {
                    float ax(x*downsize/x_factor+x_min);    // anchor location
                    float ay(y*downsize/y_factor+y_min);
                    // check all the priors
                    for (auto const &prior: priors_view) {
                        float prob = pa[0];
                        if (prob >= anchor_th) {
                            Box box;
                            box.from_residual(ax, ay, prior, pp);
                            box.s = prob;
                            boxes.append(box.make_tuple());
                        }
                        pa += 1, pp += PARAMS;
                    } // prior
                }} // x, y
                list.append(boxes);
            } // batch
            return list;
        }
    };

    /*
    py::list nms (py::list np::ndarray probs, np::ndarray params, np::ndarray priors, float anchor_th) {
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

        View<Prior> priors_view(priors);

        py::list list;
        for (int i = 0; i < batch; ++i) {
            float *pa = (float *)(probs.get_data() + probs.strides(0) * i);
            float *pp = (float *)(params.get_data() + params.strides(0) * i);
            py::list boxes;
            for (int x = 0; x < lx; ++x) {
                for (int y = 0; y < ly; ++y) {
                    float ax(x * downsize / x_factor + x_min), ay(y * downsize / y_factor + y_min);
                    // check all the priors
                    for (int k = 0; k < probs.shape(3); ++k, pa += 1, pp += n_params) {
                        float prob = pa[0];
                        if (prob < anchor_th) continue;
                        Box box;
                        box.from_residual(ax, ay, priors_view[k], pp);
                        boxes.append(box.make_tuple());
                    } // prior
                } // x
            } // y
            list.append(boxes);
        } // batch
        return list;
    }
    */

    struct Task {
        vector<string> paths;
    };

    class Streamer: public streamer::Streamer<Task>, Voxelizer {

        View<Prior> priors;
        int downsize, T;
        float lower_th, upper_th;

        Task *stage1 (py::object *obj) {
            Task *task = new Task;
            {
                streamer::ScopedGState _;
                int len = py::len(*obj);
                for (int i = 0; i < len; ++i) {
                    task->paths.push_back(py::extract<string>((*obj)[i]));
                }
                delete obj;
            }
            return task;
        }

        py::object *stage2 (Task *task) {
            /*
            void augment (np::ndarray points, np::ndarray boxes) {
                // rotate by z
                // shift by z
                std::uniform_real_distribution<float> scale(0.95, 1.05);
                std::uniform_real_distribution<float> rotate(-M_PI/4, M_PI/4);
                float s = scale(rng);
                float r = rotate(rng);
                // TODO
            }
            */
            // load from H5
            vector<vector<Point>> points;
            vector<vector<Box>> boxes;
            vector<View<Point>> points_views;
            vector<View<Box>> boxes_views;
            // load data from H5 files
            for (string const &path: task->paths) {
                H5File file(path);
                points.emplace_back();
                boxes.emplace_back();
                file.load("points", &points.back());
                file.load("boxes", &boxes.back());
                points_views.emplace_back(points.back());
                boxes_views.emplace_back(boxes.back());
            }
            delete task;

            np::ndarray *V, *M, *I;
            np::ndarray *A, *AW, *P, *PW;
            // voxelelize
            voxelize_points_helper(points_views, T, true, &V, &M, &I);
            voxelize_labels_helper(boxes_views, priors, downsize, lower_th, upper_th, true, &A, &AW, &P, &PW);
            py::object *tuple;
            {
                streamer::ScopedGState _;
                tuple = new py::tuple(py::make_tuple(py::object(), *V, *M, *I, *A, *AW, *P, *PW));
                delete V; delete M; delete I; delete A; delete AW; delete P; delete PW;
            }
            return tuple;
        }
    public:
        Streamer (py::object gen,  np::ndarray ranges, np::ndarray shape, np::ndarray priors_, int downsize_, int T_, float lower_th_, float upper_th_)
            : streamer::Streamer<Task>(gen, 6),
            Voxelizer(ranges, shape),
            priors(priors_),
            downsize(downsize_),
            T(T_),
            lower_th(lower_th_),
            upper_th(upper_th_)
        {
        }
    };
}

BOOST_PYTHON_MODULE(cpp)
{
    np::initialize();
    py::class_<Voxelizer, boost::noncopyable>("Voxelizer", py::init<np::ndarray, np::ndarray>())
        .def("voxelize_points", &Voxelizer::voxelize_points)
        .def("make_dense", &Voxelizer::make_dense)
        .def("voxelize_labels", &Voxelizer::voxelize_labels)
        .def("generate_boxes", &Voxelizer::generate_boxes)
    ;

    py::class_<Streamer, boost::noncopyable>("Streamer",
                py::init<py::object, np::ndarray, np::ndarray, np::ndarray, int, int, float, float>())
        .def("next", &Streamer::next)
    ;
}

