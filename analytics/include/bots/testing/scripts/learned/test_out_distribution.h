//
// Created by durst on 5/17/23.
//

#ifndef CSKNOW_TEST_OUT_DISTRIBUTION_H
#define CSKNOW_TEST_OUT_DISTRIBUTION_H

#include "bots/testing/scripts/learned/test_nav.h"

namespace csknow::tests::learned {
    class OutDistributionLongCTPushLongScript : public LearnedNavScript {
    public:
        explicit OutDistributionLongCTPushLongScript(size_t testIndex, size_t numTests, bool waitForever);
        void initialize(Tree & tree, ServerState & state) override;
    };

    class OutDistributionPitCTPushLongScript : public LearnedNavScript {
    public:
        explicit OutDistributionPitCTPushLongScript(size_t testIndex, size_t numTests, bool waitForever);
        void initialize(Tree & tree, ServerState & state) override;
    };

    class OutDistributionUnderACTPushLongScript : public LearnedNavScript {
    public:
        explicit OutDistributionUnderACTPushLongScript(size_t testIndex, size_t numTests, bool waitForever);
        void initialize(Tree & tree, ServerState & state) override;
    };

    class OutDistributionBRampCTPushBDoorsScript : public LearnedNavScript {
    public:
        explicit OutDistributionBRampCTPushBDoorsScript(size_t testIndex, size_t numTests, bool waitForever);
        void initialize(Tree & tree, ServerState & state) override;
    };

    class OutDistributionTopMidCTPushBDoorsScript : public LearnedNavScript {
    public:
        explicit OutDistributionTopMidCTPushBDoorsScript(size_t testIndex, size_t numTests, bool waitForever);
        void initialize(Tree & tree, ServerState & state) override;
    };

    vector<Script::Ptr> createOutDistributionNavScripts(size_t numTests, bool quitAtEnd);
}

#endif //CSKNOW_TEST_OUT_DISTRIBUTION_H
