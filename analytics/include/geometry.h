#ifndef CSKNOW_GEOMETRY_H
#define CSKNOW_GEOMETRY_H
#include <math.h>
#include <limits>
#include <iostream>
#include <vector>
#include <algorithm>
#include "load_data.h"
#include "linear_algebra.h"
//https://github.com/ValveSoftware/source-sdk-2013/blob/master/sp/src/public/mathlib/mathlib.h#L301-L303
#ifndef DEG2RAD
#define DEG2RAD( x  )  ( (double)(x) * (double)(M_PI / 180.) )
#endif
// https://counterstrike.fandom.com/wiki/Movement
#define MAX_RUN_SPEED 250.0
#define TICKS_PER_SECOND 32
#define STANDING_HEAD_FORWARD_AMOUNT 12.5
#define DUCKING_HEAD_FORWARD_AMOUNT 11.
#define STANDING_NECK_DOWN_AMOUNT 8.5
#define DUCKING_NECK_DOWN_AMOUNT 3.75
#define STANDING_HEAD_ANGLE_ADJUSTMENT_ADD 2.5
#define CROUCHING_HEAD_ANGLE_ADJUSTMENT_ADD 17.5
#define STANDING_HEAD_ANGLE_ADJUSTMENT_MUL 1.1
#define CROUCHING_HEAD_ANGLE_ADJUSTMENT_MUL 0.75

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

    string toString() {
        return "{" + orig.toString() + ", " + dir.toString() + "}";
    }
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

    void expand(double amount) {
        min = min - amount;
        max = max + amount;
    }

    string toString() {
        return "{" + min.toString() + ", " + max.toString() + "}";
    }
};

static inline __attribute__((always_inline))
bool aabbOverlap(AABB b0, AABB b1) {
    return
            (b0.min.x <= b1.max.x && b1.min.x <= b0.max.x) &&
            (b0.min.y <= b1.max.y && b1.min.y <= b0.max.y) &&
            (b0.min.z <= b1.max.z && b1.min.z <= b0.max.z);
}

static inline __attribute__((always_inline))
bool aabbOverlapExclusive(AABB b0, AABB b1) {
    return
        (b0.min.x < b1.max.x && b1.min.x < b0.max.x) &&
        (b0.min.y < b1.max.y && b1.min.y < b0.max.y) &&
        (b0.min.z < b1.max.z && b1.min.z < b0.max.z);
}

#define PLAYER_HEIGHT 72
const int STANDING_HEIGHT = 72;
const int CROUCHED_HEIGHT = 54;
const int WIDTH = 32;
const int MAX_JUMP_HEIGHT = 65;
static inline __attribute__((always_inline))
AABB getAABBForPlayer(Vec3 pos, double duckAmount = 0.) {
    //https://developer.valvesoftware.com/wiki/Counter-Strike:_Global_Offensive_Mapper%27s_Reference
    // looks like coordinates are center at feet - tested using getpos_exact and box commands from
    //https://old.reddit.com/r/csmapmakers/comments/58ch3f/useful_console_commands_for_map_making_csgo/
    //making box with these coordinates wraps player perfectly
    // https://developer.valvesoftware.com/wiki/Dimensions#Eyelevel
    // eye level is 64 units when standing, 46 when crouching
    // these numbers are weird, use mappers reference not this
    // getpos is eye level, getpos_exact is foot, both are center of model
    AABB result;
    double height = duckAmount * CROUCHED_HEIGHT + (1.-duckAmount) * STANDING_HEIGHT;
    result.min = {pos.x - WIDTH / 2, pos.y - WIDTH / 2, pos.z};
    result.max = {pos.x + WIDTH / 2, pos.y + WIDTH / 2, pos.z + height};
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

static inline __attribute__((always_inline))
bool pointInRegionMaxInclusive(AABB box, Vec3 point) {
    return point.x > box.min.x && point.x <= box.max.x &&
           point.y > box.min.y && point.y <= box.max.y &&
           point.z > box.min.z && point.z <= box.max.z;
}

static inline __attribute__((always_inline))
vector<Vec3> getAABBCorners(AABB aabb) {
    vector<Vec3> result;
    for (size_t x = 0; x < 2; x++) {
        for (size_t y = 0; y < 2; y++) {
            for (size_t z = 0; z < 2; z++) {
                result.push_back({
                    x == 0 ? aabb.min.x : aabb.max.x,
                    y == 0 ? aabb.min.y : aabb.max.y,
                    z == 0 ? aabb.min.z : aabb.max.z
                });
            }
        }
    }
    return result;
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

static inline __attribute__((always_inline))
Vec2 viewFromOriginToDest(Vec3 origin, Vec3 dest) {
    Vec3 targetVector = dest - origin;
    Vec2 targetViewAngle = vectorAngles(targetVector);
    targetViewAngle.makePitchNeg90To90();
    targetViewAngle.makeYawNeg180To180();
    return targetViewAngle;
}

static inline __attribute__((always_inline))
Vec2 deltaViewFromOriginToDest(Vec3 origin, Vec3 dest, Vec2 curViewAngle) {
    Vec3 targetVector = dest - origin;
    Vec2 targetViewAngle = vectorAngles(targetVector);
    // vectorAngles is 270 for -90, data is -90, make sure everything is noramlized before subtracting
    // so don't get into a weird range as not interested in wrapping around, want to keep values close
    targetViewAngle.makePitchNeg90To90();
    targetViewAngle.makeYawNeg180To180();
    // want to be + if too large, - if too small, so do current - target
    Vec2 deltaViewAngle = curViewAngle - targetViewAngle;
    // invert so bigger pitch is aiming up
    // DON'T DO THIS, IT MAKES CALCULATIONS A PAIN
    // deltaViewAngle.y *= -1;
    deltaViewAngle.makePitchNeg90To90();
    deltaViewAngle.makeYawNeg180To180();
    return deltaViewAngle;
}

static inline __attribute__((always_inline))
Vec2 wrappedAngleDifference(Vec2 curViewAngle, Vec2 refViewAngle) {
    // y axis is flipped, so reverse here
    Vec2 deltaViewAngle = curViewAngle - refViewAngle;
    deltaViewAngle.normalize();
    return deltaViewAngle;
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

static inline __attribute__((always_inline))
Ray getEyeCoordinatesForPlayerGivenEyeHeight(Vec3 pos, Vec2 view) {
    // same as above, but the pos.z is eye, not foot
    return Ray({pos.x, pos.y, pos.z}, angleVectors({view.x, view.y}));
}

static inline __attribute__((always_inline))
Vec3 getFootCoordinatesForPlayerGivenEyePos(Vec3 pos) {
    // same as above, but the pos.z is eye, not foot
    return {pos.x, pos.y, pos.z - EYE_HEIGHT};
}

static inline __attribute__((always_inline))
Vec3 getCenterHeadCoordinatesForPlayer(Vec3 eyePos, Vec2 view, double duckAmount) {
    // no z factor if pitch is 90 (looking down), all z factor and no x/y factor if pitch is -90 (looking up)
    // scale looking down is 0 and up is 90, perfect for sin/cos function where head makes quarter circle
    // also adjust by head angle amount since head is flat when looking down (0 deg after transformation) but back a little
    // when looking up (90 deg after transformation)
    double headAngleAdjustmentAdd = duckAmount * CROUCHING_HEAD_ANGLE_ADJUSTMENT_ADD +
            (1-duckAmount) * STANDING_HEAD_ANGLE_ADJUSTMENT_ADD;
    double headAngleAdjustmentMul = duckAmount * CROUCHING_HEAD_ANGLE_ADJUSTMENT_MUL +
                                    (1-duckAmount) * STANDING_HEAD_ANGLE_ADJUSTMENT_MUL;
    double adjustedPitch = (view.y * -1. + 90.) / 2. * headAngleAdjustmentMul + headAngleAdjustmentAdd;
    // get unit vec of just x and y (z already handled)
    Vec3 unitViewVec = angleVectors(view);
    unitViewVec.z = 0.;
    unitViewVec = unitViewVec / computeMagnitude(unitViewVec);
    double neckDownAmount = duckAmount * DUCKING_NECK_DOWN_AMOUNT + (1-duckAmount) * STANDING_NECK_DOWN_AMOUNT;
    double headForwardAmount = duckAmount * DUCKING_HEAD_FORWARD_AMOUNT + (1-duckAmount) * STANDING_HEAD_FORWARD_AMOUNT;
    return {eyePos.x + std::cos(DEG2RAD(adjustedPitch)) * unitViewVec.x * headForwardAmount,
            eyePos.y + std::cos(DEG2RAD(adjustedPitch)) * unitViewVec.y * headForwardAmount,
            eyePos.z - neckDownAmount + std::sin(DEG2RAD(adjustedPitch)) * headForwardAmount};
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
bool intersectP(const AABB & box, const Ray & ray) {
    double hitt0, hitt1;
    return intersectP(box, ray, hitt0, hitt1);
}

static inline __attribute__((always_inline))
Vec3 getCenter(AABB aabb) {
    return {(aabb.max.x + aabb.min.x)/2, (aabb.max.y + aabb.min.y)/2, (aabb.max.z + aabb.min.z)/2};
}

static inline __attribute__((always_inline))
double cosineSimilarity(Vec2 v1, Vec2 v2) {
    return v1.dot(v2) / (computeMagnitude(v1) * computeMagnitude(v2));
}

// https://www.geeksforgeeks.org/shortest-distance-between-a-line-and-a-point-in-a-3-d-plane/
static inline __attribute__((always_inline))
double computeMinDistanceLinePoint(Vec3 linePoint1, Vec3 linePoint2, Vec3 point) {
    Vec3 ab = linePoint2 - linePoint1;
    Vec3 ac = point - linePoint1;
    double area = computeMagnitude(ab * ac);
    double cd = area / computeMagnitude(ab);
    return cd;
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

static inline __attribute__((always_inline))
Vec3 translateThenRotate(Vec3 newOrigin, RotationMatrix3D originRotationMatrix, Vec3 inputPosition) {
    Vec3 translatedPosition = {inputPosition.x - newOrigin.x, inputPosition.y - newOrigin.y, inputPosition.z - newOrigin.z};
    return originRotationMatrix.rotateVec3(translatedPosition);
}

static inline __attribute__((always_inline))
double secondsAwayAtMaxSpeed(double distance) {
    return distance / MAX_RUN_SPEED;
}

#endif //CSKNOW_GEOMETRY_H
