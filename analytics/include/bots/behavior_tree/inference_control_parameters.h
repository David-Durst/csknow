//
// Created by durst on 7/26/23.
//

#ifndef CSKNOW_INFERENCE_CONTROL_PARAMETERS_H
#define CSKNOW_INFERENCE_CONTROL_PARAMETERS_H

#include "bots/analysis/feature_store_precommit.h"

namespace csknow {
    struct PlayerPushSaveControlParameters {
        bool push5s = true, push10s = true, push20s = true;

        float getPushModelValue(feature_store::DecreaseTimingOption decreaseTimingOption) const;
    };

    struct TeamSaveControlParameters {
        bool enable = true;
        bool overallPush = true;
        float temperature = 0.7;

        std::array<PlayerPushSaveControlParameters, feature_store::max_enemies>
                ctPlayerPushControlParameters, tPlayerPushControlParameters;

        float getPushModelValue(bool overall, feature_store::DecreaseTimingOption decreaseTimingOption,
                                bool ctTeam, size_t playerNum) const;
        void update(ServerState state);
    };
}


#endif //CSKNOW_INFERENCE_CONTROL_PARAMETERS_H
