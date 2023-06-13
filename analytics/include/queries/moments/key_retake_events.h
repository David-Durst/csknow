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
    class KeyRetakeEvents {
    public:
        // per tick data
        vector<bool> firedBeforeOrDuringThisTick;
        vector<bool> plantFinishedBeforeOrDuringThisTick;
        vector<bool> defusalFinishedBeforeOrDuringThisTick;
        vector<bool> explosionBeforeOrDuringThisTick;
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

        bool enableNonTestPlantRounds;

        KeyRetakeEvents(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick,
                        const Plants & plants, const Defusals & defusals, const Kills & kills,
                        const Say & say);
    };
}

#endif //CSKNOW_KEY_RETAKE_EVENTS_H
