//
// Created by durst on 6/2/23.
//

#ifndef CSKNOW_TEST_LEARNED_OVERALL_H
#define CSKNOW_TEST_LEARNED_OVERALL_H

#include "bots/testing/scripts/learned/test_learned_hold.h"
#include "bots/testing/scripts/learned/test_learned_teamwork.h"

namespace csknow::tests::learned {
    vector<Script::Ptr> createAllLearnedScripts(size_t numTests, bool quitAtEnd);
}

#endif //CSKNOW_TEST_LEARNED_OVERALL_H
