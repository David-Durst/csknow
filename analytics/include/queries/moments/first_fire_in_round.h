//
// Created by durst on 5/4/23.
//

#ifndef CSKNOW_FIRST_FIRE_IN_ROUND_H
#define CSKNOW_FIRST_FIRE_IN_ROUND_H

#include <set>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <list>
#include <map>
#include "load_data.h"

namespace csknow::first_fire {
    class FirstFireInRound {
    public:
        vector<bool> firedYet;
        FirstFireInRound(const Rounds & rounds, const Ticks & ticks);
    };
}

#endif //CSKNOW_FIRST_FIRE_IN_ROUND_H
