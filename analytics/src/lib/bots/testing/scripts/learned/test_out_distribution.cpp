//
// Created by durst on 5/17/23.
//

#include "bots/testing/scripts/learned/test_out_distribution.h"

namespace csknow::tests::learned {
    OutDistributionLongCTPushLongScript::OutDistributionLongCTPushLongScript(std::size_t testIndex, std::size_t numTests, bool waitForever) :
            LearnedNavScript("OutDistributionLongCTPushLongScript", testIndex, numTests, waitForever) { }

    void OutDistributionLongCTPushLongScript::initialize(Tree &tree, ServerState &state) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> distX(1314, 1518), distY(1369, 1895);

        LearnedNavScript::initialize(tree, state, Vec3({distX(gen), distY(gen), 95.}), Vec2({2.903987, -95.587982}),
                                     "ForceCTLong", testAToCatWaypoints, "BombsiteA");
    }

    OutDistributionPitCTPushLongScript::OutDistributionPitCTPushLongScript(std::size_t testIndex, std::size_t numTests, bool waitForever) :
            LearnedNavScript("OutDistributionPitCTPushLongScript", testIndex, numTests, waitForever) { }

    void OutDistributionPitCTPushLongScript::initialize(Tree &tree, ServerState &state) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> distX(1352, 1528), distY(436, 691);

        LearnedNavScript::initialize(tree, state, Vec3({distX(gen), distY(gen), 72.}), Vec2({2.903987, -95.587982}),
                                     "ForceCTLong", testAToCatWaypoints, "BombsiteA");
    }

    OutDistributionUnderACTPushLongScript::OutDistributionUnderACTPushLongScript(std::size_t testIndex, std::size_t numTests, bool waitForever) :
            LearnedNavScript("OutDistributionUnderACTPushLongScript", testIndex, numTests, waitForever) { }

    void OutDistributionUnderACTPushLongScript::initialize(Tree &tree, ServerState &state) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> distX(656, 912), distY(2107, 2310);

        LearnedNavScript::initialize(tree, state, Vec3({distX(gen), distY(gen), 72.}), Vec2({2.903987, -95.587982}),
                                     "ForceCTLong", testAToCatWaypoints, "BombsiteA");
    }

    OutDistributionBRampCTPushBDoorsScript::OutDistributionBRampCTPushBDoorsScript(std::size_t testIndex, std::size_t numTests, bool waitForever) :
            LearnedNavScript("OutDistributionBRampCTPushBDoorsScript", testIndex, numTests, waitForever) { }

    void OutDistributionBRampCTPushBDoorsScript::initialize(Tree &tree, ServerState &state) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> distX(-781, -586), distY(2177, 2377);

        LearnedNavScript::initialize(tree, state, Vec3({distX(gen), distY(gen), 36.}), Vec2({2.903987, -95.587982}),
                                     "ForceCTBDoors", testAToCatWaypoints, "BombsiteA");
    }

    OutDistributionTopMidCTPushBDoorsScript::OutDistributionTopMidCTPushBDoorsScript(std::size_t testIndex, std::size_t numTests, bool waitForever) :
            LearnedNavScript("OutDistributionTopMidCTPushBDoorsScript", testIndex, numTests, waitForever) { }

    void OutDistributionTopMidCTPushBDoorsScript::initialize(Tree &tree, ServerState &state) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> distX(-438, -301), distY(929, 1229);

        LearnedNavScript::initialize(tree, state, Vec3({distX(gen), distY(gen), 100.}), Vec2({2.903987, -95.587982}),
                                     "ForceCTBDoors", testAToCatWaypoints, "BombsiteA");
    }

    vector<Script::Ptr> createOutDistributionNavScripts(size_t numTests, bool quitAtEnd) {
        vector<Script::Ptr> result;

        for (size_t i = 0; i < numTests; i++) {
            result.push_back(make_unique<OutDistributionLongCTPushLongScript>(i, numTests, false));
            result.push_back(make_unique<OutDistributionPitCTPushLongScript>(i, numTests, false));
            result.push_back(make_unique<OutDistributionUnderACTPushLongScript>(i, numTests, false));
            result.push_back(make_unique<OutDistributionBRampCTPushBDoorsScript>(i, numTests, false));
            result.push_back(make_unique<OutDistributionTopMidCTPushBDoorsScript>(i, numTests, false));
        }
        if (quitAtEnd) {
            result.push_back(make_unique<QuitScript>());
        }

        return result;
    }
}
