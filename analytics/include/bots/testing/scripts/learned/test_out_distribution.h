//
// Created by durst on 5/17/23.
//

#ifndef CSKNOW_TEST_OUT_DISTRIBUTION_H
#define CSKNOW_TEST_OUT_DISTRIBUTION_H

#include "bots/testing/scripts/learned/test_learned_nav.h"

namespace csknow::tests::learned {
    class OutDistributionNavScript : public LearnedNavScript {
    public:
        std::random_device rd;
        std::mt19937 gen;
        std::uniform_real_distribution<> distX, distY;
        explicit OutDistributionNavScript(string name, size_t testIndex, size_t numTests, bool waitForever,
                                          double minX, double maxX, double minY, double maxY);
    };

    class OutDistributionLongCTPushLongScript : public OutDistributionNavScript {
    public:
        explicit OutDistributionLongCTPushLongScript(size_t testIndex, size_t numTests, bool waitForever);
        void initialize(Tree & tree, ServerState & state) override;
    };

    class OutDistributionPitCTPushLongScript : public OutDistributionNavScript {
    public:
        explicit OutDistributionPitCTPushLongScript(size_t testIndex, size_t numTests, bool waitForever);
        void initialize(Tree & tree, ServerState & state) override;
    };

    class OutDistributionUnderACTPushLongScript : public OutDistributionNavScript {
    public:
        explicit OutDistributionUnderACTPushLongScript(size_t testIndex, size_t numTests, bool waitForever);
        void initialize(Tree & tree, ServerState & state) override;
    };

    class OutDistributionBRampCTPushBDoorsScript : public OutDistributionNavScript {
    public:
        explicit OutDistributionBRampCTPushBDoorsScript(size_t testIndex, size_t numTests, bool waitForever);
        void initialize(Tree & tree, ServerState & state) override;
    };

    class OutDistributionTopMidCTPushBDoorsScript : public OutDistributionNavScript {
    public:
        explicit OutDistributionTopMidCTPushBDoorsScript(size_t testIndex, size_t numTests, bool waitForever);
        void initialize(Tree & tree, ServerState & state) override;
    };

    vector<Script::Ptr> createOutDistributionNavScripts(size_t numTests, bool quitAtEnd);
}

#endif //CSKNOW_TEST_OUT_DISTRIBUTION_H
