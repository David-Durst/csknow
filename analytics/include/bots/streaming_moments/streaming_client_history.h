//
// Created by durst on 12/25/22.
//

#ifndef CSKNOW_STREAMING_CLIENT_HISTORY_H
#define CSKNOW_STREAMING_CLIENT_HISTORY_H

#include "bots/streaming_bot_database.h"

namespace csknow {
    template <typename T>
    class StreamingClientHistory {
    public:

        unordered_map<CSGOId, CircularBuffer<T>> clientHistory;

        bool addClient(CSGOId csgoId) {
            if (clientHistory.find(csgoId) == clientHistory.end()) {
                clientHistory.insert({csgoId, CircularBuffer<T>(PAST_AIM_TICKS)});
                return true;
            }
            return false;
        }

        void removeInactiveClients(const set<CSGOId> & activeClients) {
            vector<CSGOId> historyClients;
            for (const auto & [csgoId, _] : clientHistory) {
                historyClients.push_back(csgoId);
            }

            for (const auto & clientCSGOId : historyClients) {
                if (activeClients.find(clientCSGOId) == activeClients.end()) {
                    clientHistory.erase(clientCSGOId);
                }
            }
        }
    };
}

#endif //CSKNOW_STREAMING_CLIENT_HISTORY_H
