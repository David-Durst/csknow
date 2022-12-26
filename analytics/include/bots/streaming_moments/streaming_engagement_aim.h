//
// Created by durst on 12/25/22.
//

#ifndef CSKNOW_STREAMING_ENGAGEMENT_AIM_H
#define CSKNOW_STREAMING_ENGAGEMENT_AIM_H

#include "queries/training_moments/training_engagement_aim.h"
#include "bots/streaming_bot_database.h"
#include "bots/streaming_moments/streaming_fire_history.h"
#
#include "queries/inference_moments/inference_engagement_aim.h"

namespace csknow::engagement_aim {
    class StreamingEngagementAim {
    public:
        StreamingClientHistory<EngagementAimTickData> engagementAimPlayerHistory;

        void addTickData(const StreamingBotDatabase & db,
                         const fire_history::StreamingFireHistory & streamingFireHistory);
    };
}

#endif //CSKNOW_STREAMING_ENGAGEMENT_AIM_H
