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
                                                       bool ctTeam, size_t playerNum) const {
        if (!enable) {
            return 0.5f;
        }
        else {
            if (overall) {
                return overallPush ? 1.f : 0.f;
            }
            else {
                if (ctTeam) {
                    return ctPlayerPushControlParameters[playerNum].getPushModelValue(decreaseTimingOption);
                }
                else {
                    return tPlayerPushControlParameters[playerNum].getPushModelValue(decreaseTimingOption);
                }
            }
        }
    }

    void TeamSaveControlParameters::update(ServerState state) {
        enable = state.enableAggressionControl;

    }
}