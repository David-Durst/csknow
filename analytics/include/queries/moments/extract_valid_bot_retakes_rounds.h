//
// Created by durst on 4/29/23.
//

#ifndef CSKNOW_EXTRACT_VALID_BOT_RETAKES_ROUNDS_H
#define CSKNOW_EXTRACT_VALID_BOT_RETAKES_ROUNDS_H

#include <set>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <list>
#include <map>
#include "load_data.h"
#include "queries/moments/plant_states.h"

namespace csknow::round_extractor {
    class ExtractValidBotRetakesRounds {
    public:
        vector<int64_t> validRoundIds;
        vector<int64_t> plantIndex;
        ExtractValidBotRetakesRounds(const Games & games, const Rounds & rounds);
        ExtractValidBotRetakesRounds(const csknow::plant_states::PlantStatesResult & plantStatesResult);
    };
}

#endif //CSKNOW_EXTRACT_VALID_BOT_RETAKES_ROUNDS_H
