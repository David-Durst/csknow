#ifndef CSKNOW_GEOMETRY_H
#define CSKNOW_GEOMETRY_H
#include <math.h>
#include <limits>
#include <iostream>
#include <vector>
#include "geometry.h"
#include "load_data.h"
//https://github.com/ValveSoftware/source-sdk-2013/blob/master/sp/src/public/mathlib/mathlib.h#L301-L303
#ifndef DEG2RAD
#define DEG2RAD( x  )  ( (double)(x) * (double)(M_PI / 180.) )
#endif
// https://counterstrike.fandom.com/wiki/Movement
#define MAX_RUN_SPEED 250.0
#define TICKS_PER_SECOND 32
using std::vector;

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
        dirIsNeg[0] = invdir.x < 0;
        dirIsNeg[1] = invdir.y < 0;
        dirIsNeg[2] = invdir.z < 0;
    }

    Vec3 orig, dir, invdir;
    int dirIsNeg[3];
};

struct AABB {
    Vec3 min;
    Vec3 max;

    void makeInvalid() {
        min = {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
               std::numeric_limits<double>::infinity()};
        max = {-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
               -std::numeric_limits<double>::infinity()};
    }

    void coverAllZ() {
        min.z = -std::numeric_limits<double>::infinity();
        max.z = std::numeric_limits<double>::infinity();
    }
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

static inline __attribute__((always_inline))
AABB getAABBForPlayer(Vec3 pos, double scalingFactor) {
    // for AABB that are smaller than whole bounding box, keep z a
    AABB result;
    result.min = {pos.x - WIDTH / 2 * scalingFactor, pos.y - WIDTH / 2 * scalingFactor,
                  pos.z + HEIGHT / 2 - HEIGHT / 2 * scalingFactor};
    result.max = {pos.x + WIDTH / 2 * scalingFactor, pos.y + WIDTH / 2 * scalingFactor,
                  pos.z + HEIGHT / 2 + HEIGHT / 2 * scalingFactor};
    return result;
}

static inline __attribute__((always_inline))
double computeAABBSize(AABB box) {
    double xDistance = box.max.x - box.min.x;
    double yDistance = box.max.y - box.min.y;
    double zDistance = box.max.z - box.min.z;
    return sqrt(xDistance * xDistance + yDistance * yDistance + zDistance * zDistance);
}

static inline __attribute__((always_inline))
bool pointInRegion(AABB box, Vec3 point) {
    return point.x > box.min.x && point.x < box.max.x &&
           point.y > box.min.y && point.y < box.max.y &&
           point.z > box.min.z && point.z < box.max.z;
}

// https://github.com/ValveSoftware/source-sdk-2013/blob/master/sp/src/mathlib/mathlib_base.cpp#L901-L914
static inline __attribute__((always_inline))
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

#define MachineEpsilon std::numeric_limits<double>::epsilon() * 0.5

inline constexpr double gamma2(int n) {
    return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
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
        tFar *= 1 + 2 * gamma2(3);

        t0 = tNear > t0 ? tNear : t0;
        t1 = tFar < t1 ? tFar : t1;
        if (t0 > t1)
            return false;
    }
    hitt0 = t0;
    hitt1 = t1;
    return true;
}

static inline __attribute__((always_inline))
double computeDistance(const Position &position, int source, int target, int64_t sourceIndex, int64_t targetIndex) {
    double xDistance =
            position.players[source].xPosition[sourceIndex] - position.players[target].xPosition[targetIndex];
    double yDistance =
            position.players[source].yPosition[sourceIndex] - position.players[target].yPosition[targetIndex];
    double zDistance =
            position.players[source].zPosition[sourceIndex] - position.players[target].zPosition[targetIndex];
    return sqrt(xDistance * xDistance + yDistance * yDistance + zDistance * zDistance);
}

struct CompoundAABB {
    vector<AABB> regions;
};

static inline __attribute__((always_inline))
bool pointInCompoundRegion(CompoundAABB boxes, Vec3 point) {
    for (const auto & box : boxes) {
        if (pointInRegion(box, point)) {
            return true;
        }
    }
    return false;
}

#endif //CSKNOW_GEOMETRY_H
