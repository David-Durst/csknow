//
// Created by durst on 12/25/22.
//

#ifndef CSKNOW_STREAMING_MANAGER_H
#define CSKNOW_STREAMING_MANAGER_H

#include <bots/streaming_bot_database.h>
#include <bots/streaming_moments/streaming_fire_history.h>
#include <bots/streaming_moments/streaming_engagement_aim.h>
#include <bots/streaming_moments/streaming_test_logger.h>

class StreamingManager {
public:
    StreamingBotDatabase db;
    csknow::test_log::StreamingTestLogger streamingTestLogger;
    csknow::fire_history::StreamingFireHistory streamingFireHistory;
    csknow::engagement_aim::StreamingEngagementAim streamingEngagementAim;
    bool forceReset = false;

    StreamingManager(const string & navPath) : streamingTestLogger(navPath), streamingEngagementAim(navPath)  { }
    void update(const ServerState & state);
};

#endif //CSKNOW_STREAMING_MANAGER_H
