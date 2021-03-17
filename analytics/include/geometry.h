#ifndef CSKNOW_GEOMETRY_H
#define CSKNOW_GEOMETRY_H
#include <math.h>
#include <limits>
#include <iostream>
#include "geometry.h"
//https://github.com/ValveSoftware/source-sdk-2013/blob/master/sp/src/public/mathlib/mathlib.h#L301-L303
#ifndef DEG2RAD
#define DEG2RAD( x  )  ( (double)(x) * (double)(M_PI / 180.) )
#endif

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

const int HEIGHT = 72;
const int WIDTH = 32;
static inline __attribute__((always_inline))
AABB getAABBForPlayer(Vec3 pos) {
    //https://developer.valvesoftware.com/wiki/Counter-Strike:_Global_Offensive_Mapper%27s_Reference
    // looks like coordinates are center at feet - tested using getpos_exact and box commands from
    //https://old.reddit.com/r/csmapmakers/comments/58ch3f/useful_console_commands_for_map_making_csgo/
    //making box with these coordinates wraps player perfectly
    AABB result;
    result.min = {pos.x - WIDTH / 2, pos.y - WIDTH / 2, pos.z};
    result.max = {pos.x + WIDTH / 2, pos.y + WIDTH / 2, pos.z + HEIGHT};
    return result;
}

// https://github.com/ValveSoftware/source-sdk-2013/blob/master/sp/src/mathlib/mathlib_base.cpp#L901-L914
Vec3 angleVectors(const Vec2 &angles) {
    Vec3 forward;
    double sp, sy, cp, cy;

    sincos( DEG2RAD( angles.x ), &sy, &cy );
    sincos( DEG2RAD( angles.y ), &sp, &cp );

    forward.x = cp*cy;
    forward.y = cp*sy;
    forward.z = -sp;
    return forward;
}

const int EYE_HEIGHT = 64;
static inline __attribute__((always_inline))
Ray getEyeCoordinatesForPlayer(Vec3 pos, Vec2 view) {
    //https://developer.valvesoftware.com/wiki/Counter-Strike:_Global_Offensive_Mapper%27s_Reference
    // looks like coordinates are center at feet - tested using getpos_exact and box commands from
    //https://old.reddit.com/r/csmapmakers/comments/58ch3f/useful_console_commands_for_map_making_csgo/
    // don't negative view.y as it runs negative (it's pitch, not actually y), used in angleVectors
    return Ray({pos.x, pos.y, pos.z + EYE_HEIGHT}, angleVectors({view.x, view.y}));
}


// https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/util/pstd.h#L26-L31
template <typename T>
static inline __attribute__((always_inline))
void swap(T &a, T &b) {
    T tmp = std::move(a);
    a = std::move(b);
    b = std::move(tmp);
}

// https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/util/vecmath.h#L1555
static inline __attribute__((always_inline))
bool intersectP(const AABB & box, const Ray & ray, double & hitt0, double & hitt1,
                double tMax = std::numeric_limits<double>::infinity()) {
    double t0 = 0, t1 = tMax;
    for (int i = 0; i < 3; ++i) {
        // Update interval for _i_th bounding box slab
        double invRayDir = 1 / ray.dir[i];
        double tNear = (box.min[i] - ray.orig[i]) * invRayDir;
        double tFar = (box.max[i] - ray.orig[i]) * invRayDir;
        // Update parametric interval from slab intersection $t$ values
        if (tNear > tFar)
            swap(tNear, tFar);
        // Update _tFar_ to ensure robust ray--bounds intersection
        tFar *= 1 + 2 * gamma(3);

        t0 = tNear > t0 ? tNear : t0;
        t1 = tFar < t1 ? tFar : t1;
        std::cout << "t0: " << t0 << std::endl;
        std::cout << "t1: " << t1 << std::endl;
        if (t0 > t1)
            return false;
    }
    hitt0 = t0;
    hitt1 = t1;
    return true;
}

#endif //CSKNOW_GEOMETRY_H
