//
// Created by durst on 12/25/22.
//

#ifndef CSKNOW_STREAMING_ENGAGEMENT_AIM_H
#define CSKNOW_STREAMING_ENGAGEMENT_AIM_H

#include "queries/training_moments/training_engagement_aim.h"
#include "bots/streaming_bot_database.h"
#include "bots/streaming_moments/streaming_fire_history.h"
#include "queries/inference_moments/inference_engagement_aim.h"

namespace csknow::engagement_aim {
    struct EngagementAimTarget {
        // if set to invalid, then need to make a target from a position
        CSGOId csgoId;
        Vec3 pos;

        bool isPlayer() const {
            return csgoId != INVALID_ID;
        }

        bool operator==(const EngagementAimTarget & other) const {
            return csgoId == other.csgoId && pos == other.pos;
        }

        bool operator!=(const EngagementAimTarget & other) const {
            return csgoId != other.csgoId || pos != other.pos;
        }
    };

    typedef unordered_map<CSGOId, EngagementAimTarget> ClientTargetMap;

    class StreamingEngagementAim {
        EngagementAimTickData computeOneTickData(StreamingBotDatabase & db,
                                                 const fire_history::StreamingFireHistory & streamingFireHistory,
                                                 CSGOId attackerId, const EngagementAimTarget & target,
                                                 size_t attackerStateOffset, size_t victimStateOffset,
                                                 bool firstEngagementTick, const VisPoints & visPoints);
        void predictNewAngles(const StreamingBotDatabase & db);
    public:
        StreamingClientHistory<EngagementAimTickData> engagementAimPlayerHistory;

        ClientTargetMap currentClientTargetMap, priorClientTargetMap;
        unordered_map<CSGOId, StreamingPinId> playerToVictimLastAlivePos;
        unordered_map<CSGOId, Vec3> playerToVictimEngagementFirstHeadPos;
        unordered_map<CSGOId, bool> playerToVictimFirstVisibleFrame;
        unordered_map<CSGOId, Vec2> playerToNewAngles;

        void addTickData(StreamingBotDatabase & db,
                         const fire_history::StreamingFireHistory & streamingFireHistory,
                         const VisPoints & visPoints);
    };
}

#endif //CSKNOW_STREAMING_ENGAGEMENT_AIM_H
