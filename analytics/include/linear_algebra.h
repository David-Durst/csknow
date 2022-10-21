//
// Created by durst on 3/28/22.
//

#ifndef CSKNOW_LINEAR_ALGEBRA_H
#define CSKNOW_LINEAR_ALGEBRA_H
#include <cmath>
#include <vector>
#include <array>
using std::vector;
using std::array;

#define DEG2RAD(angleInDegrees) ((angleInDegrees) * M_PI / 180.0)
#define RAD2DEG(angleInRadians) ((angleInRadians) * 180.0 / M_PI)


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

    bool operator==(const Vec3& rhs)
    {
        return x == rhs.x && y == rhs.y && z == rhs.z;
    }

    bool operator!=(const Vec3& rhs)
    {
        return x != rhs.x || y != rhs.y || z == rhs.z;
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

    Vec3 operator*(const Vec3 & other) const {
        Vec3 result;
        result.x = y * other.z - z * other.y;
        result.y = z * other.x - x * other.z;
        result.z = x * other.y - y * other.x;
        return result;
    }

    Vec3 operator+(double offset) const {
        Vec3 result = *this;
        result.x += offset;
        result.y += offset;
        result.z += offset;
        return result;
    }

    Vec3 operator-(double offset) const {
        Vec3 result = *this;
        result.x -= offset;
        result.y -= offset;
        result.z -= offset;
        return result;
    }

    Vec3 operator*(double scale) const {
        Vec3 result = *this;
        result.x *= scale;
        result.y *= scale;
        result.z *= scale;
        return result;
    }

    Vec3 operator/(double scale) const {
        Vec3 result = *this;
        result.x /= scale;
        result.y /= scale;
        result.z /= scale;
        return result;
    }

    string toString() const {
        return "{" + std::to_string(x) + ", " + std::to_string(y)
               + ", " + std::to_string(z) + "}";
    }
    string toCSV() const {
        return std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z);
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

static inline __attribute__((always_inline))
double computeDistance(Vec3 v1, Vec3 v2) {
    double xDistance = v1.x - v2.x;
    double yDistance = v1.y - v2.y;
    double zDistance = v1.z - v2.z;
    return sqrt(xDistance * xDistance + yDistance * yDistance + zDistance * zDistance);
}

static inline __attribute__((always_inline))
double computeMagnitude(Vec3 v) {
    return computeDistance(v, {0, 0, 0});
}

static inline __attribute__((always_inline))
Vec3 unitize(Vec3 v) {
    return v / computeMagnitude(v);
}

static inline __attribute__((always_inline))
IVec3 vec3ToIVec3(Vec3 a) {
    return {static_cast<int64_t>(a.x), static_cast<int64_t>(a.y), static_cast<int64_t>(a.z)};
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

    Vec2 operator/(double scale) const {
        Vec2 result = *this;
        result.x /= scale;
        result.y /= scale;
        return result;
    }

    Vec2 operator*(double scale) const {
        Vec2 result = *this;
        result.x *= scale;
        result.y *= scale;
        return result;
    }

    double dot(const Vec2 & other) const {
        return (this->x * other.x) + (this->y * other.y);
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

    void normalizeYawPitchRelativeToOther(Vec2 other) {
        if (x > other.x + 180.) {
            x -= 360.;
        }
        else if (x + 180. < other.x) {
            x += 360.;
        }
        if (y > other.y + 90.) {
            //y -= 180.;
        }
        else if (y + 90. < other.y) {
            //y += 180.;
        }
    }

    Vec2 & normalize() {
        makeYawNeg180To180();
        makePitchNeg90To90();
        return *this;
    }

    string toString() {
        return "{" + std::to_string(x) + ", " + std::to_string(y) + "}";
    }

    string toCSV() const {
        return std::to_string(x) + "," + std::to_string(y);
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

static inline __attribute__((always_inline))
double computeMagnitude(Vec2 v) {
    return computeDistance({v.x, v.y, 0}, {0, 0, 0});
}

static inline __attribute__((always_inline))
Vec2 unitize(Vec2 v) {
    return v / computeMagnitude(v);
}

struct RotationMatrix3D {
    array<array<double, 3>, 3> values;

    // https://danceswithcode.net/engineeringnotes/rotations_in_3d/rotations_in_3d_part2.html
    RotationMatrix3D(Vec2 yawPitch, bool fromViewAngle) {
        // if it's from another player's view angles, then invert angles as want rotation so other angles relative to that one
        double invertFactor = fromViewAngle ? -1. : 1.;
        double yaw = yawPitch.x * invertFactor;
        double pitch = yawPitch.y * invertFactor;
        double roll = 0;
        //Precompute sines and cosines of Euler angles
        double su = std::sin(roll);
        double cu = std::cos(roll);
        double sv = std::sin(pitch);
        double cv = std::cos(pitch);
        double sw = std::sin(yaw);
        double cw = std::cos(yaw);

        //Create and populate RotationMatrix
        values[0][0] = cv*cw;
        values[0][1] = su*sv*cw - cu*sw;
        values[0][2] = su*sw + cu*sv*cw;
        values[1][0] = cv*sw;
        values[1][1] = cu*cw + su*sv*sw;
        values[1][2] = cu*sv*sw - su*cw;
        values[2][0] = -sv;
        values[2][1] = su*cv;
        values[2][2] = cu*cv;
    }

    Vec3 rotateVec3(Vec3 vec) {
        Vec3 result;
        result.x = vec.x * values[0][0] + vec.y * values[0][1] + vec.z * values[0][2];
        result.y = vec.x * values[1][0] + vec.y * values[1][1] + vec.z * values[1][2];
        result.z = vec.x * values[2][0] + vec.y * values[2][1] + vec.z * values[2][2];
        return result;
    }
};

#endif //CSKNOW_LINEAR_ALGEBRA_H
