//
// Created by durst on 1/5/23.
//

#include <bots/analysis/streaming_manager.h>

void StreamingManager::update(const ServerState & state) {
    // if any client teleports, clear everyone's history
    for (const auto & client : state.clients) {
        if (client.lastTeleportId != client.lastTeleportConfirmationId || forceReset) {
            db.clear();
            streamingEngagementAim.reset();
            forceReset = false;
            streamingEngagementAim.aimTicksFile << "reset" << std::endl;
            break;
        }
    }

    streamingTestLogger.setCurFrameTime();
    if (streamingTestLogger.testActive()) {
        for (size_t i = 0; i < state.weaponFireEvents.size(); i++) {
            streamingTestLogger.addEvent("weapon fire");
        }
        for (size_t i = 0; i < state.hurtEvents.size(); i++) {
            streamingTestLogger.addEvent("hurt");
        }
    }

    db.addState(state);
    streamingFireHistory.addTickData(db);
    streamingEngagementAim.addTickData(db, streamingFireHistory);
}
