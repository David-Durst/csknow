//
// Created by durst on 1/5/23.
//

#include <bots/analysis/streaming_manager.h>

void StreamingManager::update(const ServerState & state) {
    // if any client teleports, clear everyone's history
    for (const auto & client : state.clients) {
        if (client.lastTeleportId != client.lastTeleportConfirmationId) {
            db.clear();
            streamingEngagementAim.reset();
            break;
        }
    }

    db.addState(state);
    streamingFireHistory.addTickData(db);
    streamingEngagementAim.addTickData(db, streamingFireHistory);
}
