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
            streamingTestLogger.addEvent("weapon fire", "");
        }
        for (size_t i = 0; i < state.hurtEvents.size(); i++) {
            streamingTestLogger.addEvent("hurt", "");
        }
    }

    db.addState(state);
    streamingFireHistory.addTickData(db);
    streamingEngagementAim.addTickData(db, streamingFireHistory);

    if (streamingTestLogger.testActive() && streamingTestLogger.attackerId != INVALID_ID) {
        const EngagementAimTickData & attackerEngagementAimTickData =
            streamingEngagementAim.engagementAimPlayerHistory.clientHistory.at(streamingTestLogger.attackerId).fromNewest();
        streamingTestLogger.addEvent("angular distance x",
                                     std::to_string(attackerEngagementAimTickData.deltaRelativeCurHeadViewAngle.x));
        streamingTestLogger.addEvent("angular distance y",
                                     std::to_string(attackerEngagementAimTickData.deltaRelativeCurHeadViewAngle.y));
        streamingTestLogger.addEvent("angular target min x",
                                     std::to_string(attackerEngagementAimTickData.victimRelativeFirstHeadMinViewAngle.x));
        streamingTestLogger.addEvent("angular target min y",
                                     std::to_string(attackerEngagementAimTickData.victimRelativeFirstHeadMinViewAngle.y));
        streamingTestLogger.addEvent("angular target max x",
                                     std::to_string(attackerEngagementAimTickData.victimRelativeFirstHeadMaxViewAngle.x));
        streamingTestLogger.addEvent("angular target max y",
                                     std::to_string(attackerEngagementAimTickData.victimRelativeFirstHeadMaxViewAngle.y));
    }
}
