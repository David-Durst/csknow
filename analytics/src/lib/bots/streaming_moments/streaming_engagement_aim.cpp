//
// Created by durst on 12/25/22.
//

#include "bots/streaming_moments/streaming_engagement_aim.h"

namespace csknow::engagement_aim {
    void StreamingEngagementAim::addTickData(const StreamingBotDatabase & db,
                                             const fire_history::StreamingFireHistory & streamingFireHistory) {
        const ServerState & curState = db.batchData.fromNewest();
        set<CSGOId> activeClients;

        for (const auto & curTickClient : curState.clients) {
            activeClients.insert(curTickClient.csgoId);
            const fire_history::FireClientData & curFireClientData =
                streamingFireHistory.fireClientHistory.clientHistory.at(curTickClient.csgoId).fromNewest();
            EngagementAimTickData engagementAimTickData;


            engagementAimPlayerHistory.clientHistory.at(curTickClient.csgoId).enqueue(engagementAimTickData);
        }

        engagementAimPlayerHistory.removeInactiveClients(activeClients);
    }
}