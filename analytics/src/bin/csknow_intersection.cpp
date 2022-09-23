#ifdef false
#include "geometry.h"
#include <iostream>

int main(int argc, char * argv[]) {
    Vec3 eyePos = {40.45, -558.37, 65.46};
    Vec2 eyeAngle = {43.82, 3.1};
    Vec3 eyeView = {0.720463, 0.691382, -0.0540788};
    Vec3 victim = {735.23, 554.34, 1.27};
    Vec3 victimMin = {719.23, 538.34, 1.27};
    Vec3 victimMax = { 751.23, 570.34, 73.27};
    Ray cheaterRay(eyePos, eyeView);
    AABB victimBox = {victimMin, victimMax};
    double hitt0, hitt1;
    bool result = intersectP(victimBox, cheaterRay, hitt0, hitt1);
    std::cout << "result: " << result << std::endl;
    std::cout << "hitt0 " << hitt0 << std::endl;
    std::cout << "hitt1 " << hitt1 << std::endl;
    std::cout << "t0 x " << cheaterRay.orig[0] + cheaterRay.dir[0] * hitt0
        << ", y " << cheaterRay.orig[1] + cheaterRay.dir[1] * hitt0
        << ", z " << cheaterRay.orig[2] + cheaterRay.dir[2] * hitt0 << std::endl;
    std::cout << "t1 x " << cheaterRay.orig[0] + cheaterRay.dir[0] * hitt1
              << ", y " << cheaterRay.orig[1] + cheaterRay.dir[1] * hitt1
              << ", z " << cheaterRay.orig[2] + cheaterRay.dir[2] * hitt1 << std::endl;
}
#endif