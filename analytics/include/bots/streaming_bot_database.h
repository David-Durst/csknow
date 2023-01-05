//
// Created by durst on 12/25/22.
//

#ifndef CSKNOW_STREAMING_BOT_DATABASE_H
#define CSKNOW_STREAMING_BOT_DATABASE_H

#include <bots/load_save_bot_data.h>
#include <queries/training_moments/training_engagement_aim.h>
#include <load_data.h>
#include <circular_buffer.h>
#define STREAMING_HISTORY_TICKS 20
typedef int64_t StreamingPinId;

class StreamingBotDatabase {
public:
    CircularBuffer<ServerState> batchData;
    unordered_map<StreamingPinId, ServerState> pinnedData;
    StreamingPinId nextPinId;


    StreamingBotDatabase() : batchData(STREAMING_HISTORY_TICKS), nextPinId(0) { }

    void clear() {
        pinnedData.clear();
        batchData.clear();
        nextPinId = 0;
    }

    void addState(const ServerState & state) {
        batchData.enqueue(state);
    }

    StreamingPinId pinState(size_t ticksFromPresent = 0) {
        pinnedData[nextPinId] = batchData.fromNewest(static_cast<int64_t>(ticksFromPresent));
        nextPinId++;
        return nextPinId - 1;
    }

    void unpinState(StreamingPinId pinId) {
        pinnedData.erase(pinId);
        if (pinnedData.empty()) {
            nextPinId = 0;
        }
    }

    size_t clientHistoryLength(CSGOId csgoId) const {
        for (int64_t i = 0; i < STREAMING_HISTORY_TICKS; i++) {
            // hit history length if no more overall history or no more player-specific histroy
            if (batchData.getCurSize() <= i ||
                batchData.fromNewest(i).csgoIds.find(csgoId) == batchData.fromNewest(i).csgoIds.end()) {
                return static_cast<size_t>(i);
            }
        }

        return STREAMING_HISTORY_TICKS;
    }
};

#endif //CSKNOW_STREAMING_BOT_DATABASE_H
