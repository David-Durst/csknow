//
// Created by durst on 7/26/23.
//

#include "bots/behavior_tree/inference_control_parameters.h"

namespace csknow {
    float PlayerPushSaveControlParameters::getPushModelValue(feature_store::DecreaseTimingOption decreaseTimingOption)
        const {
        bool param = false;
        if (decreaseTimingOption == feature_store::DecreaseTimingOption::s5) {
            param = push5s;
        }
        else if (decreaseTimingOption == feature_store::DecreaseTimingOption::s10) {
            param = push10s;
        }
        else if (decreaseTimingOption == feature_store::DecreaseTimingOption::s20) {
            param = push20s;
        }
        return param ? 1.f : 0.f;
    }

    float TeamSaveControlParameters::getPushModelValue(bool overall,
                                                       feature_store::DecreaseTimingOption decreaseTimingOption,
                                                       CSGOId playerId) const {
        if (!enable) {
            return 0.5f;
        }
        else {
            if (overall) {
                return overallPush ? 1.f : 0.f;
            }
            else {
                if (playerPushControlParameters.count(playerId) == 0) {
                    throw std::runtime_error("invalid player id for push control parameters");
                }
                return playerPushControlParameters.at(playerId).getPushModelValue(decreaseTimingOption);
            }
        }
    }

    void TeamSaveControlParameters::update(const ServerState & state) {
        enable = state.enableAggressionControl;
        overallPush = state.pushRound;
        playerPushControlParameters.clear();
        for (const auto & client : state.clients) {
            playerPushControlParameters[client.csgoId] = {client.push5s, client.push10s, client.push20s};
        }
    }
}