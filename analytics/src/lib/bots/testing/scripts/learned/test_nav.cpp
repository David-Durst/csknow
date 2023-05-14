//
// Created by durst on 5/14/23.
//

#include "bots/testing/scripts/learned/test_nav.h"

namespace csknow::tests::learned {
    LearnedGooseToCatScript::LearnedGooseToCatScript(size_t testIndex, size_t numTests, bool waitForever) :
            Script("LearnedGooseToCatScript", {{0, ENGINE_TEAM_CT}}, {ObserveType::FirstPerson, 0}),
            testIndex(testIndex), numTests(numTests), waitForever(waitForever) {
        name += "," + std::to_string(testIndex) + "," + std::to_string(numTests);
    }

    void LearnedGooseToCatScript::initialize(Tree &tree, ServerState &state) {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            set<AreaId> areasToRemove;
            Node::Ptr setupCommands = make_unique<SequenceNode>(blackboard, Node::makeList(
                make_unique<InitTestingRound>(blackboard, name),
                make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id},state),
                make_unique<SetPos>(blackboard, Vec3({1071.936035, 2972.308837, 128.762023}), Vec2({-84.903987, -95.587982})),
                make_unique<Teleport>(blackboard, neededBots[0].id, state),
                make_unique<ForceOrderNode>(blackboard, "ForceTCat", vector{neededBots[0].id}, testAToCatWaypoints, areasToRemove, addedOrderId),
                make_unique<StandingStill>(blackboard, vector{neededBots[0].id})),
            "LearnedGooseToLongSetup");

            Node::Ptr disableAllBothDuringSetup = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                    std::move(setupCommands),
                    make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[0].id}, false)
            ), "DisableDuringSetup");

            Node::Ptr finishCondition;
            if (waitForever) {
                finishCondition = make_unique<RepeatDecorator>(
                        blackboard, make_unique<InPlace>(blackboard, neededBots[0].id, "Catwalk"), true);
            }
            else {
                finishCondition = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                    make_unique<RepeatDecorator>(blackboard, make_unique<InPlace>(blackboard, neededBots[0].id, "Catwalk"), true),
                    make_unique<movement::WaitNode>(blackboard, 30, false)));
            }

            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                std::move(disableAllBothDuringSetup),
                make_unique<SayCmd>(blackboard, test_ready_string),
                std::move(finishCondition)),
            "LearnedGooseToLongSequence");
        }
    }

    vector<Script::Ptr> createLearnedNavScripts(size_t numTests, bool quitAtEnd) {
        vector<Script::Ptr> result;

        for (size_t i = 0; i < numTests; i++) {
            result.push_back(make_unique<LearnedGooseToCatScript>(i, numTests, false));
        }
        if (quitAtEnd) {
            result.push_back(make_unique<QuitScript>());
        }

        return result;
    }
}