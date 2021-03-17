#ifndef CSKNOW_GEOMETRY_H
#define CSKNOW_GEOMETRY_H

struct Vec3 {
    double x;
    double y;
    double z;

    double operator[](size_t index) const {
        if (index == 0) {
            return x;
        } else if (index == 1) {
            return y;
        } else {
            return z;
        }
    }
};

struct Vec2 {
    double x;
    double y;
};

class Ray {
public:
    Ray() { }
    Ray(const Vec3 &orig, const Vec3 &dir) : orig(orig), dir(dir) {
        invdir.x = 1 / dir.x;
        invdir.y = 1 / dir.y;
        invdir.z = 1 / dir.y;
        sign[0] = invdir.x < 0;
        sign[1] = invdir.y < 0;
        sign[2] = invdir.z < 0;
    }

    Vec3 orig, dir, invdir;
    int sign[3];
};

struct AABB {
    Vec3 min;
    Vec3 max;
};

static inline __attribute__((always_inline))
AABB getAABBForPlayer(Vec3 pos);

Vec3 angleVectors(const Vec2 &angles);


static inline __attribute__((always_inline))
Ray getEyeCoordinatesForPlayer(Vec3 pos, Vec2 view);


template <typename T>
static inline __attribute__((always_inline))
void swap(T &a, T &b);


static inline __attribute__((always_inline))
bool intersectP(const AABB & box, const Ray & ray, double * hitt0, double * hitt1,
                double tMax = std::numeric_limits<double>::infinity());

#endif //CSKNOW_GEOMETRY_H
