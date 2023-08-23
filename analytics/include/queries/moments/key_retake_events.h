//
// Created by durst on 5/4/23.
//

#ifndef CSKNOW_KEY_RETAKE_EVENTS_H
#define CSKNOW_KEY_RETAKE_EVENTS_H

#include <set>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <list>
#include <map>
#include "load_data.h"

namespace csknow::key_retake_events {
    constexpr int max_enemies = 5;

    struct PerTraceData {
        // trace data
        vector<string> demoFile;
        vector<int> traceIndex;
        vector<int> numTraces;
        // loading names in key retake event, converted to index-based bools in feature store team
        vector<vector<string>> nonReplayPlayers;
        vector<bool> convertedNonReplayNamesToIndices;
        std::array<vector<bool>, max_enemies> ctIsBotPlayer, tIsBotPlayer;
        vector<bool> oneNonReplayTeam;
        vector<bool> oneNonReplayBot;
    };

    class KeyRetakeEvents {
    public:
        // per tick data
        vector<bool> firedBeforeOrDuringThisTick;
        vector<bool> plantFinishedBeforeOrDuringThisTick;
        vector<bool> defusalFinishedBeforeOrDuringThisTick;
        vector<bool> explosionBeforeOrDuringThisTick;
        vector<bool> ctAlive;
        vector<bool> tAlive;
        vector<bool> ctAliveAfterExplosion;
        vector<bool> tAliveAfterDefusal;
        // per tick data tracking test states
        vector<bool> testStartBeforeOrDuringThisTick;
        vector<bool> testEndBeforeOrDuringThisTick;

        // per round data
        vector<bool> roundHasPlant;
        vector<int> roundCTAliveOnPlant;
        vector<int> roundTAliveOnPlant;
        vector<bool> roundHasDefusal;
        vector<bool> roundHasRetakeCTSave;
        vector<bool> roundHasRetakeTSave;
        vector<bool> roundHasRetakeSave;
        vector<int> roundC4Deaths;
        vector<int> roundNonC4PostPlantWorldDeaths;
        // test state data
        vector<string> roundTestName;
        vector<int> roundTestIndex;
        vector<int> roundNumTests;
        vector<bool> roundHasStartTest;
        vector<bool> roundHasCompleteTest;
        vector<bool> roundHasFailedTest;
        vector<bool> roundBaiters;

        PerTraceData perTraceData;


        bool enableNonTestPlantRounds;

        KeyRetakeEvents(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick,
                        const Plants & plants, const Defusals & defusals, const Kills & kills,
                        const Say & say);
    };
}

#endif //CSKNOW_KEY_RETAKE_EVENTS_H
