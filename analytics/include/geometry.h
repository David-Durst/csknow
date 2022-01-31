#ifndef CSKNOW_GEOMETRY_H
#define CSKNOW_GEOMETRY_H
#include <math.h>
#include <limits>
#include <iostream>
#include <vector>
#include <algorithm>
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

struct IVec3 {
    int64_t x;
    int64_t y;
    int64_t z;

    bool operator==(const IVec3& rhs)
    {
        return x == rhs.x && y == rhs.y && z == rhs.z;
    }

    bool operator!=(const IVec3& rhs)
    {
        return x != rhs.x || y != rhs.y || z == rhs.z;
    }

    IVec3 operator+(int64_t value) const {
        IVec3 result = *this;
        result.x += value;
        result.y += value;
        result.z += value;
        return result;
    }

    IVec3 operator-(int64_t value) const {
        IVec3 result = *this;
        result.x -= value;
        result.y -= value;
        result.z -= value;
        return result;
    }
};

static inline __attribute__((always_inline))
IVec3 min(IVec3 a, IVec3 b) {
    IVec3 result;
    result.x = std::min(a.x, b.x);
    result.y = std::min(a.y, b.y);
    result.z = std::min(a.z, b.z);
    return result;
}

static inline __attribute__((always_inline))
IVec3 max(IVec3 a, IVec3 b) {
    IVec3 result;
    result.x = std::max(a.x, b.x);
    result.y = std::max(a.y, b.y);
    result.z = std::max(a.z, b.z);
    return result;
}

static inline __attribute__((always_inline))
double positiveModulo(double x, double y) {
    return fmod(fmod(x, y) + y, y);
}

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

    Vec3 operator+(const Vec3 & other) const {
        Vec3 result = *this;
        result.x += other.x;
        result.y += other.y;
        result.z += other.z;
        return result;
    }

    Vec3 operator-(const Vec3 & other) const {
        Vec3 result = *this;
        result.x -= other.x;
        result.y -= other.y;
        result.z -= other.z;
        return result;
    }
};

static inline __attribute__((always_inline))
Vec3 min(Vec3 a, Vec3 b) {
    Vec3 result;
    result.x = std::min(a.x, b.x);
    result.y = std::min(a.y, b.y);
    result.z = std::min(a.z, b.z);
    return result;
}

static inline __attribute__((always_inline))
Vec3 max(Vec3 a, Vec3 b) {
    Vec3 result;
    result.x = std::max(a.x, b.x);
    result.y = std::max(a.y, b.y);
    result.z = std::max(a.z, b.z);
    return result;
}

struct Vec2 {
    double x;
    double y;

    double& operator[](size_t index) {
        if (index == 0) {
            return x;
        } else {
            return y;
        }
    }

    Vec2 operator+(const Vec2 & other) const {
        Vec2 result = *this;
        result.x += other.x;
        result.y += other.y;
        return result;
    }

    Vec2 operator-(const Vec2 & other) const {
        Vec2 result = *this;
        result.x -= other.x;
        result.y -= other.y;
        return result;
    }

    void makeYawNeg180To180() {
        this->x = positiveModulo(this->x, 360.);
        if (this->x > 180.) {
            this->x -= 360.;
        }
    }

    void makePitch0To360() {
        this->y = positiveModulo(this->y, 360.);
        if (this->y < 0.) {
            this->y += 360.;
        }
    }

    void makePitchNeg90To90() {
        this->y = positiveModulo(this->y, 360.);
        if (this->y > 260.) {
            this->y -= 360.;
        }
    }
};

static inline __attribute__((always_inline))
Vec2 min(Vec2 a, Vec2 b) {
    Vec2 result;
    result.x = std::min(a.x, b.x);
    result.y = std::min(a.y, b.y);
    return result;
}

static inline __attribute__((always_inline))
Vec2 max(Vec2 a, Vec2 b) {
    Vec2 result;
    result.x = std::max(a.x, b.x);
    result.y = std::max(a.y, b.y);
    return result;
}

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

static inline __attribute__((always_inline))
bool aabbOverlap(AABB b0, AABB b1) {
    return
            (b0.min.x <= b1.max.x && b1.min.x <= b0.max.x) &&
            (b0.min.y <= b1.max.y && b1.min.y <= b0.max.y) &&
            (b0.min.z <= b1.max.z && b1.min.z <= b0.max.z);
}

const int HEIGHT = 72;
const int WIDTH = 32;
static inline __attribute__((always_inline))
AABB getAABBForPlayer(Vec3 pos) {
    //https://developer.valvesoftware.com/wiki/Counter-Strike:_Global_Offensive_Mapper%27s_Reference
    // looks like coordinates are center at feet - tested using getpos_exact and box commands from
    //https://old.reddit.com/r/csmapmakers/comments/58ch3f/useful_console_commands_for_map_making_csgo/
    //making box with these coordinates wraps player perfectly
    // https://developer.valvesoftware.com/wiki/Dimensions#Eyelevel
    // eye level is 64 units when standing, 46 when crouching
    // getpos is eye level, getpos_exact is foot, both are center of model
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
// https://developer.valvesoftware.com/wiki/QAngle - QAngle is just a regular Euler angle
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

// https://github.com/ValveSoftware/source-sdk-2013/blob/master/sp/src/mathlib/mathlib_base.cpp#L1000-L1029
static inline __attribute__((always_inline))
Vec2 vectorAngles(const Vec3 &forward)
{
    Vec2 angles;
	float tmp, yaw, pitch;
	
	if (forward[1] == 0 && forward[0] == 0)
	{
		yaw = 0;
		if (forward[2] > 0)
			pitch = 270;
		else
			pitch = 90;
	}
	else
	{
		yaw = (atan2(forward[1], forward[0]) * 180 / M_PI);
		if (yaw < 0)
			yaw += 360;

		tmp = sqrt(forward[0]*forward[0] + forward[1]*forward[1]);
		pitch = (atan2(-forward[2], tmp) * 180 / M_PI);
		if (pitch < 0)
			pitch += 360;
	}
	
	angles[0] = yaw;
	angles[1] = pitch;
    return angles;
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
            std::swap(tNear, tFar);
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
Vec3 getCenter(AABB aabb) {
    return {(aabb.max.x + aabb.min.x)/2, (aabb.max.y + aabb.min.y)/2, (aabb.max.z + aabb.min.z)/2};
}

static inline __attribute__((always_inline))
double computeDistance(Vec3 v1, Vec3 v2) {
    double xDistance = v1.x - v2.x;
    double yDistance = v1.y - v2.y;
    double zDistance = v1.z - v2.z;
    return sqrt(xDistance * xDistance + yDistance * yDistance + zDistance * zDistance);
}

static inline __attribute__((always_inline))
double computeDistance(const PlayerAtTick & playerAtTick, int64_t sourceIndex, int64_t targetIndex) {
    double xDistance =
            playerAtTick.posX[sourceIndex] - playerAtTick.posX[targetIndex];
    double yDistance =
            playerAtTick.posY[sourceIndex] - playerAtTick.posY[targetIndex];
    double zDistance =
            playerAtTick.posZ[sourceIndex] - playerAtTick.posZ[targetIndex];
    return sqrt(xDistance * xDistance + yDistance * yDistance + zDistance * zDistance);
}

struct CompoundAABB {
    vector<AABB> regions;
};

static inline __attribute__((always_inline))
bool pointInCompoundRegion(CompoundAABB boxes, Vec3 point) {
    for (const auto & box : boxes.regions) {
        if (pointInRegion(box, point)) {
            return true;
        }
    }
    return false;
}

#endif //CSKNOW_GEOMETRY_H
