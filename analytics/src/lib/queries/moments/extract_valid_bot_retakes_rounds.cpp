//
// Created by durst on 4/29/23.
//
#include "queries/moments/extract_valid_bot_retakes_rounds.h"

namespace csknow::round_extractor {
    ExtractValidBotRetakesRounds::ExtractValidBotRetakesRounds(const Games & games, const Rounds & rounds) {
        for (int64_t gameIndex = 0; gameIndex < games.size; gameIndex++) {
            vector<int64_t> winRounds;
            int64_t lastWinRound = INVALID_ID;
            for (int64_t roundIndex = games.roundsPerGame[gameIndex].minId;
                 roundIndex <= games.roundsPerGame[gameIndex].maxId; roundIndex++) {
                if (rounds.winner[roundIndex] == ENGINE_TEAM_T || rounds.winner[roundIndex] == ENGINE_TEAM_CT) {
                    winRounds.push_back(roundIndex);
                    lastWinRound = roundIndex;
                }
            }

            // first win round is for triggering the start, last win round is for reset, so get all between
            for (size_t i = 1; i < winRounds.size(); i++) {
                if (winRounds[i] == lastWinRound) {
                    continue;
                }
                validRoundIds.push_back(winRounds[i]);
                plantIndex.push_back(i-1);
            }

            if (gameIndex == 0) {
                std::cout << "valid rounds and plant index for game 0:" << std::endl;
                for (size_t i = 0; i < validRoundIds.size(); i++) {
                    std::cout << validRoundIds[i] << "," << plantIndex[i] << std::endl;
                }
            }
        }
    }

    ExtractValidBotRetakesRounds::ExtractValidBotRetakesRounds(
        const csknow::plant_states::PlantStatesResult & plantStatesResult) {
        for (int64_t i = 0; i < plantStatesResult.size; i++) {
            validRoundIds.push_back(plantStatesResult.roundId[i]);
            plantIndex.push_back(i);
        }

    }
}