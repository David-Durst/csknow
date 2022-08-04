//
// Created by durst on 8/3/22.
//

#ifndef CSKNOW_SECOND_ORDER_CONTROLLER_H
#define CSKNOW_SECOND_ORDER_CONTROLLER_H

#include "geometry.h"
#include <optional>
using std::optional;

// https://www.youtube.com/watch?v=KPoeNZZ6H4s (13:26)
class SecondOrderController {
    Vec2 xp; // previous input
    Vec2 yd; // state variables
    double k1, k2, k3; // dynamic constants

public:
    SecondOrderController(double f, double z, double r, Vec2 x0 = {0., 0.}) {
        // compute constants
        k1 = z / (M_PI * f);
        k2 = 1 / ((2 * M_PI * f) * (2 * M_PI *f));
        k3 = r * z / (2 * M_PI * f);
        // initialize variables
        xp = x0;
        yd = {0., 0.};
    }

    Vec2 update(double T, Vec2 x, Vec2 yp, optional<Vec2> xd = {}) {
        if (!xd) { // estimate velocity
            xd = (x - xp) / T;
            xp = x;
        }
        double k2_stable = std::max(k2, 1.1 * (T*T/4 + T*k1/2));
        Vec2 y = yp + yd * T;
        yd = yd + (x + xd.value()*k3 - y - yd*k1) * T / k2_stable;
        return y;
    }
};


#endif //CSKNOW_SECOND_ORDER_CONTROLLER_H
