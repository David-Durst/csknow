//
// Created by durst on 12/25/22.
//

#ifndef CSKNOW_STREAMING_MANAGER_H
#define CSKNOW_STREAMING_MANAGER_H

#include <bots/streaming_bot_database.h>
#include <bots/streaming_moments/streaming_fire_history.h>

class StreamingManager {
public:
    StreamingBotDatabase db;
    csknow::fire_history::StreamingFireHistory streamingFireHistory;

    void update(const ServerState & state) {
        db.addState(state);
        streamingFireHistory.addTickData(db);
    }

};

#endif //CSKNOW_STREAMING_MANAGER_H
