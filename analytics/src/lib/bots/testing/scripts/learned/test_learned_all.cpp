//
// Created by durst on 6/2/23.
//
#include "bots/testing/scripts/learned/test_learned_all.h"

namespace csknow::tests::learned {
    vector<Script::Ptr> createAllLearnedScripts(size_t numTests, bool quitAtEnd) {
        vector<Script::Ptr> result;

        for (size_t i = 0; i < numTests; i++) {
            result.push_back(make_unique<LearnedHoldASitePushScript>(i, numTests, false));
            result.push_back(make_unique<LearnedHoldASiteBaitScript>(i, numTests, false));
            result.push_back(make_unique<LearnedHoldBSitePushScript>(i, numTests, false));
            result.push_back(make_unique<LearnedHoldBSiteBaitScript>(i, numTests, false));
            result.push_back(make_unique<LearnedPushLurkBaitASiteScript>(i, numTests, false));
            result.push_back(make_unique<LearnedPushATwoOrdersScript>(i, numTests, false));
            result.push_back(make_unique<LearnedPushThreeBScript>(i, numTests, false));
        }
        if (quitAtEnd) {
            result.push_back(make_unique<QuitScript>());
        }

        return result;
    }
}