#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <boost/ref.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/adapted/c_array.hpp>
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

    typedef boost::geometry::model::d2::point_xy<float> point_xy;
    typedef boost::geometry::model::polygon<point_xy> Polygon;

    static int PARAMS = 8;
    int random_seed = 2019;
    static float constexpr PIx2 = M_PI * 2;

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
        static int constexpr DIM = 5;
        union {
            float data[5];
            struct {
                float l, w, h, z, qt; // length is along x, width is along y
            };
        };

        float t () const {
            return qt * M_PI / 2;
        }
        // qt = 0 1 2 -1
    };

    static float intersect (float min1, float max1, float min2, float max2) {
        // intersection length of range [min1, max1] and [min2, max2]
        float a = std::max(min1, min2);
        float b = std::min(max1, max2);
        if (a <= b) return b-a;
        return 0;
    }

    float norm_angle (float d) {
        if (d >= PIx2) d -= PIx2;
        else if (d <= -PIx2) d += PIx2;
        float ad = d >= 0 ? (d - PIx2) : (d + PIx2);
        if (abs(ad) < abs(d)) return ad;
        return d;
    }

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
        bool sanity_check (Prior const &p, float ax, float ay) const {
            do {
                if (h <= 0) break;
                if (w <= 0) break;
                if (l <= 0) break;
                if (h > 2) break;
                if (w > 2) break;
                if (l > 5) break;
                float d = sqrt(w * w + l * l);
                float dist = sqrt((x-ax)*(x-ax) + (y-ay)*(y-ay));
                if (d < dist) break;
                return true;
            } while (false);
            std::cerr << "BAD " << ax << ' ' << ay << ' ' << h << ' ' << w << ' ' << l << std::endl;
            return false;
        }

        void from_prior (Prior const &p, float ax, float ay) {
            x = ax;
            y = ay;
            z = p.z;
            h = p.h;
            w = p.w;
            l = p.l;
            t = p.t();
            s = 0;
        }

        void augment (float sc, float dx, float dy, float dz, float dt, float s, float c) {
            // s: sin(dt)
            // c: cos(dt)
            float tx = x * sc + dx;
            float ty = y * sc + dy;
            x = c * tx - s * ty;
            y = s * tx + c * ty;
            z = z * sc + dz;
            h *= sc;
            w *= sc;
            l *= sc;
            t += dt;
        }

        void load (float const *params) {
            std::copy(params, params + DIM, data);
        }

        void store (float *params) {
            std::copy(data, data + DIM, params);
        }

        py::tuple make_tuple () const {
            return py::make_tuple(x, y, z, h, w, l, t, s);
        }

        void from_tuple (py::tuple box) {
            x = py::extract<float>(box[0]);
            y = py::extract<float>(box[1]);
            z = py::extract<float>(box[2]);
            h = py::extract<float>(box[3]);
            w = py::extract<float>(box[4]);
            l = py::extract<float>(box[5]);
            t = py::extract<float>(box[6]);
            s = py::extract<float>(box[7]);
        }

        void to_residual (float ax, float ay, Prior const &prior, float *params) const {
            // residual is the regression target
            float d = sqrt(prior.l * prior.l + prior.w * prior.w);

            params[0] = (x - ax);
            params[1] = (y - ay);
            CHECK(abs(params[1]) < 8);
            params[2] = (z - prior.z) / prior.z;
            params[3] = (h - prior.h) / prior.h;
            params[4] = (w - prior.w) / prior.w;
            params[5] = (l - prior.l) / prior.l;
            params[6] = norm_angle(t - prior.t()) / M_PI;
            params[7] = 0;
        }

        void from_residual (float ax, float ay, Prior const &prior, float const *params) {
            float d = sqrt(prior.l * prior.l + prior.w * prior.w);
            CHECK(d < 10);

            x = params[0] + ax;
            y = params[1] + ay;
            z = params[2] * prior.z + prior.z;
            h = params[3] * prior.h + prior.h;
            w = params[4] * prior.w + prior.w;
            l = params[5] * prior.l + prior.l;
            t = params[6] * M_PI + prior.t();
        }

#if 0
        float score_anchor (float ax, float ay, Prior const &prior) const {
            // approximate
            float a = l * l + w * w;
            float d = sqrt(a)/2;
            float a2 = prior.l * prior.l + prior.w * prior.w;
            float d2 = sqrt(a2)/2;
            float i = intersect(x-d, x+d, ax-d2, ax+d2)   // intersection area
                     * intersect(y-d, y+d, ay-d2, ay+d2);
            return i / (a + a2 - i + 0.00001);
        }
#endif

		void polygon (Polygon *poly) const {
			namespace bl = boost::numeric::ublas;
			using namespace boost::geometry;
			bl::matrix<float> mref(2, 2);
            // we are using -t to calculate the matrix
            // this is determined by visualization
			mref(0, 0) = cos(t); mref(0, 1) = -sin(t);
			mref(1, 0) = sin(t); mref(1, 1) = cos(t);

			bl::matrix<float> corners(2, 4);
			float data[] = {w / 2, w / 2, -w / 2, -w / 2,
							l / 2, -l / 2, -l / 2, l / 2};
			std::copy(data, data + 8, corners.data().begin());
			bl::matrix<float> gc = prod(mref, corners);
			for (int i = 0; i < 4; ++i) {
				gc(0, i) += x;
				gc(1, i) += y;
			}

			float points[][2] = {{gc(0, 0), gc(1, 0)},{gc(0, 1), gc(1, 1)},{gc(0, 2), gc(1, 2)},{gc(0, 3), gc(1, 3)},{gc(0, 0), gc(1, 0)}};
			boost::geometry::append(*poly, points);
		}
    };

#if 0
    float iou (Box const &a, Box const &b) {
        // approximate
        float aa = a.x * a.x + a.y * a.y;
        float ab = b.x * b.x + b.y * b.y;   // area
        float ra = sqrt(aa)/2;
        float rb = sqrt(ab)/2;
        float i = intersect(a.x-ra, a.x+ra, b.x-rb, b.x+rb)
                * intersect(a.y-ra, a.y+ra, b.y-rb, b.y+rb);
        return i / (aa + ab - i + 0.00001);
    }
#endif

    float iou (Polygon const &p1, Polygon const &p2) {
        vector<Polygon> in, un;
        boost::geometry::intersection(p1, p2, in);
        boost::geometry::union_(p1, p2, un);
        float inter_area = in.empty() ? 0 : boost::geometry::area(in.front());
        float union_area = boost::geometry::area(un.front());
        return inter_area / union_area;
    }

    // Give unmanaged memory an array-like accessment interface.
    template <typename T>
    class View {
        T *data;
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

        View (vector<T> &v) {
            data = &v[0];
            sz = v.size();
        }

        T &operator [] (size_t i) {
            return data[i];
        }

        T const &operator [] (size_t i) const {
            return data[i];
        }
        size_t size () const { return sz; }

        bool empty () const { return sz == 0; }

        T const *begin () const { return data; }
        T const *end () const { return data + sz; }
        T *begin () { return data; }
        T *end () { return data + sz; }
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
    protected:
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

        void augment_helper (vector<View<Point>> &points_batch, vector<View<Box>> &boxes_batch, int seed) {
            // TODO: per-box rotation
            CHECK(points_batch.size() == boxes_batch.size());
            std::default_random_engine rng1(seed);
            std::uniform_real_distribution<float> shiftxy(-0.5, 0.5);
            std::uniform_real_distribution<float> shiftz(-0.2, 0.2);
            std::uniform_real_distribution<float> shift2(-0.05, 0.05);
            std::uniform_real_distribution<float> scale(0.95, 1.05);
            std::uniform_real_distribution<float> rotate(-M_PI/4, M_PI/4);
            //vector<float> r;
            for (unsigned i = 0; i < points_batch.size(); ++i) {
                float sc = scale(rng);
                float dx = shiftxy(rng), dy = shiftxy(rng), dz = shiftz(rng);
                float dt = rotate(rng);

                View<Point> &points = points_batch[i];
                View<Box> &boxes = boxes_batch[i];
                //r.resize(points.size());
                float s = sin(dt), c = cos(dt);

                for (unsigned j = 0; j < points.size(); ++j) {
                    Point &point = points[j];
                    float x = point.x * sc + dx + shift2(rng);
                    float y = point.y * sc + dy + shift2(rng);
                    point.x = c * x - s * y;
                    point.y = s * x + c * y;
                    point.z = point.z * sc + dz + shift2(rng);
                }

                for (Box &box: boxes) {
                    box.augment(sc, dx, dy, dz, dt, s, c);
                }
            }
        }

        void augment (py::list points_batch, py::list boxes_batch) {
            int batch = py::len(boxes_batch);
            CHECK(batch == py::len(points_batch));
            vector<View<Point>> points_views;
            vector<View<Box>> boxes_views;
            for (int i = 0; i < batch; ++i) {
                np::ndarray points = py::extract<np::ndarray>(points_batch[i]);
                check_dense<float>(points, 2);
                CHECK(points.shape(1) == 4);
                points_views.emplace_back(points);

                // foreach batch
                np::ndarray boxes = py::extract<np::ndarray>(boxes_batch[i]);
                check_dense<float>(boxes, 2);
                CHECK(boxes.shape(1) == 8);
                boxes_views.emplace_back(boxes);
            }
            augment_helper(points_views, boxes_views, rng());
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
                I_buf = alloc_ndarray<int32_t>(py::make_tuple(N, 1), I);
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
                vector<Polygon> box_polygons(boxes.size());
                for (unsigned i = 0; i < boxes.size(); ++i) {
                    boxes[i].polygon(&box_polygons[i]);
                }
                for (int x = 0; x < lx; ++x) { for (int y = 0; y < ly; ++y) {
                    float ax(x*downsize/x_factor+x_min), ay(y*downsize/y_factor+y_min);
                    for (auto const &prior: priors) {
                        Box prior_box;
                        Polygon prior_polygon;
                        prior_box.from_prior(prior, ax, ay);
                        prior_box.polygon(&prior_polygon);
                        Box const *best_box = nullptr;
                        float best_d = 0;
                        for (unsigned j = 0; j < boxes.size(); ++j) {
                            float d = iou(prior_polygon, box_polygons[j]);
                            float dt = abs(norm_angle(boxes[j].t - prior.t()));
                            if (dt > 3 * M_PI/8) d = 0;
                            if (d > best_d) {   // find best circle
                                best_d = d;
                                best_box = &boxes[j];
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
                }}} // prior, y, x
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
                            if (box.sanity_check(prior, ax, ay)) {
                                boxes.append(box.make_tuple());
                            }
                        }
                        pa += 1, pp += PARAMS;
                    } // prior
                }} // x, y
                list.append(boxes);
            } // batch
            return list;
        }

        py::list box2polygon (py::tuple b) {
            Box box;
            box.from_tuple(b);
            Polygon poly;
            box.polygon(&poly);
            py::list list;
            for (auto const &p: poly.outer()) {
                float x = (p.x() - x_min) * x_factor;
                float y = (p.y() - y_min) * y_factor;
                list.append(py::make_tuple(x, y));
            }
            return list;
        }
    };

    py::list nms (py::list inputs, float nms_th) {
        py::list outputs;
        for (int i = 0; i < len(inputs); ++i) {
            vector<Box> boxes;
            {
                py::list list = py::extract<py::list>(inputs[i]);
                for (int j = 0; j < len(list); ++j) {
                    Box box;
                    box.from_tuple(py::extract<py::tuple>(list[j]));
                    boxes.push_back(box);
                }
            }
            std::sort(boxes.begin(), boxes.end(), [](Box const &a, Box const &b) { return a.s > b.s; });

            vector<Box> keep;
            vector<Polygon> polygons;
            for (auto const &box: boxes) {
                Polygon polygon;
                box.polygon(&polygon);
                bool good = true;
                for (auto const &polygon2: polygons) {

                    if (iou(polygon, polygon2) >= nms_th) {
                        good = false;
                        break;
                    }
                }
                if (good) {
                    keep.push_back(box);
                    polygons.push_back(polygon);
                }
            }

            {
                py::list list;
                for (auto const &box: keep) {
                    list.append(box.make_tuple());
                }
                outputs.append(list);
            }
        } 
        return outputs;
    }

    struct Task {
        py::object *inputs;
        vector<string> paths;
        int seed;
    };

    class Streamer: public streamer::Streamer<Task>, Voxelizer {

        View<Prior> priors;
        int downsize, T;
        float lower_th, upper_th;

        Task *stage1 (py::object *obj) {
            Task *task = new Task;
            task->seed = rng();
            task->inputs = obj;
            {
                streamer::ScopedGState _;
                int len = py::len(*obj);
                for (int i = 0; i < len; ++i) {
                    task->paths.push_back(py::extract<string>((*obj)[i]));
                }
            }
            return task;
        }

        py::object *stage2 (Task *task) {
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
            augment_helper(points_views, boxes_views, task->seed);
            /*
                if (boxes_views.back().size()) {
                float min_z = boxes_views.back()[0].z;
                float max_z = min_z;
                for (Box const &b: boxes_views.back()) {
                    min_z = std::min(min_z, b.z);
                    max_z = std::max(max_z, b.z);
                }
                std::cerr << "MINMAX " << min_z << " " << max_z << std::endl;
                }
            */

            np::ndarray *V, *M, *I;
            np::ndarray *A, *AW, *P, *PW;
            // voxelelize
            voxelize_points_helper(points_views, T, true, &V, &M, &I);
            voxelize_labels_helper(boxes_views, priors, downsize, lower_th, upper_th, true, &A, &AW, &P, &PW);
            py::object *tuple;
            {
                streamer::ScopedGState _;
                tuple = new py::tuple(py::make_tuple(*task->inputs, *V, *M, *I, *A, *AW, *P, *PW));
                delete task->inputs;
                delete V; delete M; delete I; delete A; delete AW; delete P; delete PW;
            }
            delete task;
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
        .def("voxelize_labels", &Voxelizer::voxelize_labels)
        .def("generate_boxes", &Voxelizer::generate_boxes)
        .def("box2polygon", &Voxelizer::box2polygon)
        .def("augment", &Voxelizer::augment)
    ;

    py::class_<Streamer, boost::noncopyable>("Streamer",
                py::init<py::object, np::ndarray, np::ndarray, np::ndarray, int, int, float, float>())
        .def("next", &Streamer::next)
    ;

    def("nms", ::nms);
}

