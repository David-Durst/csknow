//
// Created by durst on 12/25/22.
//

#ifndef CSKNOW_STREAMING_MANAGER_H
#define CSKNOW_STREAMING_MANAGER_H

#include <bots/streaming_bot_database.h>
#include <bots/streaming_moments/streaming_fire_history.h>
#include <bots/streaming_moments/streaming_engagement_aim.h>

class StreamingManager {
public:
    StreamingBotDatabase db;
    csknow::fire_history::StreamingFireHistory streamingFireHistory;
    csknow::engagement_aim::StreamingEngagementAim streamingEngagementAim;

    StreamingManager(const string & navPath) : streamingEngagementAim(navPath) { }

    void update(const ServerState & state, const VisPoints & visPoints) {
        db.addState(state);
        streamingFireHistory.addTickData(db);
        streamingEngagementAim.addTickData(db, streamingFireHistory, visPoints);
    }

};

#endif //CSKNOW_STREAMING_MANAGER_H
