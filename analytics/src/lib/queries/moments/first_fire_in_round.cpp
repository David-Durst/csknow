//
// Created by durst on 5/4/23.
//

#include "queries/moments/first_fire_in_round.h"

namespace csknow::first_fire {
    FirstFireInRound::FirstFireInRound(const Rounds &rounds, const Ticks & ticks) {
        firedYet.resize(ticks.size, false);
        for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
            bool foundFireYet = false;
            for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
                 tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {
                if (!foundFireYet) {
                    for (const auto & [_0, _1, fireIndex] :
                            ticks.weaponFirePerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                        foundFireYet = true;
                        break;
                    }
                }
                firedYet[tickIndex] = foundFireYet;
            }
        }
    }
}
