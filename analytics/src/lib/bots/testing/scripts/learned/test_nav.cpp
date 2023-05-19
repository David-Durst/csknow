//
// Created by durst on 5/14/23.
//

#include "bots/testing/scripts/learned/test_nav.h"
#include "bots/analysis/learned_models.h"

namespace csknow::tests::learned {
    LearnedNavScript::LearnedNavScript(const std::string & name, size_t testIndex, size_t numTests, bool waitForever) :
            Script(name, {{0, ENGINE_TEAM_CT}}, {ObserveType::FirstPerson, 0}),
            testIndex(testIndex), numTests(numTests), waitForever(waitForever) { }

    void LearnedNavScript::initialize(Tree & tree, ServerState & state, Vec3 startPos, Vec2 startViewAngle,
                                      const std::string & forceOrderNodeName, const vector<Waypoint> & waypoints,
                                      const std::string & destinationPlace) {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            set<AreaId> areasToRemove;
            Node::Ptr setupCommands = make_unique<SequenceNode>(blackboard, Node::makeList(
                make_unique<InitTestingRound>(blackboard, name),
                make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id},state),
                make_unique<SetPos>(blackboard, startPos, startViewAngle),
                make_unique<Teleport>(blackboard, neededBots[0].id, state),
                make_unique<ForceOrderNode>(blackboard, forceOrderNodeName, vector{neededBots[0].id}, waypoints, areasToRemove, addedOrderId),
                make_unique<movement::WaitNode>(blackboard, 0.1),
                make_unique<RepeatDecorator>(blackboard,
                                             make_unique<StandingStill>(blackboard, vector{neededBots[0].id}), true)),
            "Setup");

            Node::Ptr disableAllBothDuringSetup = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                    std::move(setupCommands),
                    make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[0].id}, false)
            ), "DisableDuringSetup");

            Node::Ptr placeCheck;
            if (getPlaceAreaModelProbabilities(ENGINE_TEAM_CT) || getPlaceAreaModelProbabilities(ENGINE_TEAM_T)) {
                placeCheck = make_unique<NearPlace>(blackboard, neededBots[0].id, destinationPlace, 200);
            }
            else {
                placeCheck = make_unique<InPlace>(blackboard, neededBots[0].id, destinationPlace);
            }

            Node::Ptr finishCondition;
            if (waitForever) {
                finishCondition = make_unique<RepeatDecorator>(blackboard, std::move(placeCheck), true);
            }
            else {
                finishCondition = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                    make_unique<RepeatDecorator>(blackboard, std::move(placeCheck), true),
                    make_unique<movement::WaitNode>(blackboard, 30, false)));
            }

            string specificDetails = name + "," + std::to_string(testIndex) + "," + std::to_string(numTests);
            string specificTestReadyString = test_ready_string + "," + specificDetails;
            string specificTestFinishedString = test_finished_string + "," + specificDetails;
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                std::move(disableAllBothDuringSetup),
                make_unique<SayCmd>(blackboard, specificTestReadyString),
                std::move(finishCondition),
                make_unique<SayCmd>(blackboard, specificTestFinishedString)),
            "LearnedNavSequence");
        }
    }

    LearnedGooseToCatScript::LearnedGooseToCatScript(std::size_t testIndex, std::size_t numTests, bool waitForever) :
            LearnedNavScript("LearnedGooseToCatScript", testIndex, numTests, waitForever) { }

    void LearnedGooseToCatScript::initialize(Tree &tree, ServerState &state) {
        LearnedNavScript::initialize(tree, state, Vec3({1071.936035, 2972.308837, 128.762023}),
                                     Vec2({-84.903987, -95.587982}), "ForceTCat", testAToCatWaypoints, "Catwalk");
    }

    LearnedCTPushLongScript::LearnedCTPushLongScript(std::size_t testIndex, std::size_t numTests, bool waitForever) :
            LearnedNavScript("LearnedCTPushLongScript", testIndex, numTests, waitForever) { }

    void LearnedCTPushLongScript::initialize(Tree &tree, ServerState &state) {
        LearnedNavScript::initialize(tree, state, Vec3({593., 282., 2.}), Vec2({2.903987, -95.587982}),
                                     "ForceCTLong", strategy::offenseLongToAWaypoints, "BombsiteA");
    }

    LearnedCTPushBDoorsScript::LearnedCTPushBDoorsScript(std::size_t testIndex, std::size_t numTests, bool waitForever) :
            LearnedNavScript("LearnedCTPushBDoorsScript", testIndex, numTests, waitForever) { }

    void LearnedCTPushBDoorsScript::initialize(Tree &tree, ServerState &state) {
        LearnedNavScript::initialize(tree, state, Vec3({-516., 1733., -14.}), Vec2({-89.683349, 0.746031}),
                                     "ForceBDoors", strategy::offenseBDoorsToBWaypoints, "BombsiteB");
    }

    LearnedCTPushBHoleScript::LearnedCTPushBHoleScript(std::size_t testIndex, std::size_t numTests, bool waitForever) :
            LearnedNavScript("LearnedCTPushBHoleScript", testIndex, numTests, waitForever) { }

    void LearnedCTPushBHoleScript::initialize(Tree &tree, ServerState &state) {
        LearnedNavScript::initialize(tree, state, Vec3({-516., 1733., -14.}), Vec2({-89.683349, 0.746031}),
                                     "ForceBHole", strategy::offenseHoleToBWaypoints, "BombsiteB");
    }

    vector<Script::Ptr> createLearnedNavScripts(size_t numTests, bool quitAtEnd) {
        vector<Script::Ptr> result;

        for (size_t i = 0; i < numTests; i++) {
            //result.push_back(make_unique<LearnedGooseToCatScript>(i, numTests, false));
            //result.push_back(make_unique<LearnedCTPushLongScript>(i, numTests, false));
            result.push_back(make_unique<LearnedCTPushBDoorsScript>(i, numTests, false));
            result.push_back(make_unique<LearnedCTPushBHoleScript>(i, numTests, false));
        }
        if (quitAtEnd) {
            result.push_back(make_unique<QuitScript>());
        }

        return result;
    }
}