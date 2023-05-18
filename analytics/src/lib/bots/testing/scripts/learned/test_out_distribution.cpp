//
// Created by durst on 5/17/23.
//

#include "bots/testing/scripts/learned/test_out_distribution.h"

namespace csknow::tests::learned {
    OutDistributionNavScript::OutDistributionNavScript(string name, size_t testIndex, size_t numTests, bool waitForever,
                                                       double minX, double maxX, double minY, double maxY) :
            LearnedNavScript(name, testIndex, numTests, waitForever),
            gen(rd()), distX(minX, maxX), distY(minY, maxY) { }

    OutDistributionLongCTPushLongScript::OutDistributionLongCTPushLongScript(std::size_t testIndex, std::size_t numTests, bool waitForever) :
            OutDistributionNavScript("OutDistributionLongCTPushLongScript", testIndex, numTests, waitForever,
                                     1314., 1518., 1369., 1895.)  { }

    void OutDistributionLongCTPushLongScript::initialize(Tree &tree, ServerState &state) {
        LearnedNavScript::initialize(tree, state, Vec3({distX(gen), distY(gen), 95.}), Vec2({2.903987, -95.587982}),
                                     "ForceCTLong", testAToCatWaypoints, "BombsiteA");
    }

    OutDistributionPitCTPushLongScript::OutDistributionPitCTPushLongScript(std::size_t testIndex, std::size_t numTests, bool waitForever) :
            OutDistributionNavScript("OutDistributionPitCTPushLongScript", testIndex, numTests, waitForever,
                             1352., 1528., 436., 691.) { }

    void OutDistributionPitCTPushLongScript::initialize(Tree &tree, ServerState &state) {
        OutDistributionNavScript::initialize(tree, state, Vec3({distX(gen), distY(gen), 72.}), Vec2({2.903987, -95.587982}),
                                     "ForceCTLong", testAToCatWaypoints, "BombsiteA");
    }

    OutDistributionUnderACTPushLongScript::OutDistributionUnderACTPushLongScript(std::size_t testIndex, std::size_t numTests, bool waitForever) :
            OutDistributionNavScript("OutDistributionUnderACTPushLongScript", testIndex, numTests, waitForever,
                                     656., 912., 2107., 2310.) { }

    void OutDistributionUnderACTPushLongScript::initialize(Tree &tree, ServerState &state) {
        OutDistributionNavScript::initialize(tree, state, Vec3({distX(gen), distY(gen), 72.}), Vec2({2.903987, -95.587982}),
                                     "ForceCTLong", testAToCatWaypoints, "BombsiteA");
    }

    OutDistributionBRampCTPushBDoorsScript::OutDistributionBRampCTPushBDoorsScript(std::size_t testIndex, std::size_t numTests, bool waitForever) :
            OutDistributionNavScript("OutDistributionBRampCTPushBDoorsScript", testIndex, numTests, waitForever,
                                     -781., -586., 2177., 2377.) { }

    void OutDistributionBRampCTPushBDoorsScript::initialize(Tree &tree, ServerState &state) {
        OutDistributionNavScript::initialize(tree, state, Vec3({distX(gen), distY(gen), 36.}), Vec2({2.903987, -95.587982}),
                                     "ForceBDoors", strategy::offenseBDoorsToBWaypoints, "BombsiteB");
    }

    OutDistributionTopMidCTPushBDoorsScript::OutDistributionTopMidCTPushBDoorsScript(std::size_t testIndex, std::size_t numTests, bool waitForever) :
            OutDistributionNavScript("OutDistributionTopMidCTPushBDoorsScript", testIndex, numTests, waitForever,
                                     -438., -301., 929., 1229.) { }

    void OutDistributionTopMidCTPushBDoorsScript::initialize(Tree &tree, ServerState &state) {
        LearnedNavScript::initialize(tree, state, Vec3({distX(gen), distY(gen), 100.}), Vec2({2.903987, -95.587982}),
                                     "ForceBDoors", strategy::offenseBDoorsToBWaypoints, "BombsiteB");
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
