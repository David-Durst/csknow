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

    if (streamingTestLogger.testActive()) {
        for (const auto & weaponFireEvent : state.weaponFireEvents) {
            streamingTestLogger.addEvent("weapon fire");
        }
        for (const auto & hurt : state.hurtEvents) {
            streamingTestLogger.addEvent("hurt");
        }
    }

    db.addState(state);
    streamingFireHistory.addTickData(db);
    streamingEngagementAim.addTickData(db, streamingFireHistory);
}
