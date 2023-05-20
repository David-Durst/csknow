//
// Created by durst on 5/18/23.
//

#include "bots/testing/scripts/learned/test_hold.h"
#include "bots/testing/scripts/learned/log_nodes.h"

namespace csknow::tests::learned {
    LearnedHoldScript::LearnedHoldScript(
            const std::string &name, vector<NeededBot> neededBots, ObserveSettings observeSettings,
            std::size_t testIndex, std::size_t numTests, bool waitForever) :
            Script(name, neededBots, observeSettings),
                   testIndex(testIndex), numTests(numTests), waitForever(waitForever) { }

    void LearnedHoldScript::initialize(Tree &tree, ServerState &state, vector<Vec3> playerPos,
                                       vector<Vec2> playerViewAngle, Vec3 c4Pos, vector<AreaId> chokeAreas) {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            set<AreaId> areasToRemove;
            vector<CSGOId> neededBotIds = getNeededBotIds();
            Node::Ptr setupCommands = make_unique<SequenceNode>(blackboard, Node::makeList(
                    make_unique<InitTestingRound>(blackboard, name),
                    make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                    make_unique<SlayAllBut>(blackboard, neededBotIds, state),
                    make_unique<TeleportMultiple>(blackboard, neededBotIds, playerPos, playerViewAngle, state),
                    make_unique<SetPos>(blackboard, c4Pos, Vec2({0., 0.})),
                    make_unique<TeleportPlantedC4>(blackboard),
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    make_unique<RepeatDecorator>(blackboard,
                                                 make_unique<StandingStill>(blackboard, neededBotIds), true),
                    make_unique<RecomputeOrdersNode>(blackboard)),
                "Setup");

            Node::Ptr disableAllBothDuringSetup = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                    std::move(setupCommands),
                    make_unique<DisableActionsNode>(blackboard, "DisableSetup", neededBotIds, false)
            ), "DisableDuringSetup");

            Node::Ptr stillAndLookingAtChoke = make_unique<RepeatDecorator>(blackboard,
                    make_unique<SequenceNode>(blackboard, Node::makeList(
                            make_unique<StandingStill>(blackboard, neededBotIds),
                            make_unique<AimingAtArea>(blackboard, neededBotIds, chokeAreas[0]),
                            make_unique<AimingAtArea>(blackboard, neededBotIds, chokeAreas[1]),
                            make_unique<AimingAtArea>(blackboard, neededBotIds, chokeAreas[2]))),
                    true);

            /*
            Node::Ptr toggleLearnedControllers = make_unique<SequenceNode>(blackboard, Node::makeList(
                            make_unique<SayCmd>(blackboard, "disabling learned controllers"),
                            make_unique<DisableLearnedControllers>(blackboard),
                            make_unique<movement::WaitNode>(blackboard, 7, true),
                            make_unique<SayCmd>(blackboard, "enabling learned controllers"),
                            make_unique<EnableLearnedControllers>(blackboard),
                            make_unique<movement::WaitNode>(blackboard, 40, true)),
                    "ToggleLearnedControllers");
            */

            Node::Ptr finishCondition;
            if (waitForever) {
                finishCondition = make_unique<RepeatDecorator>(blackboard, std::move(stillAndLookingAtChoke), true);
            }
            else {
                finishCondition = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                        //std::move(toggleLearnedControllers),
                        make_unique<RepeatDecorator>(blackboard, std::move(stillAndLookingAtChoke), true),
                        make_unique<FailIfTimeoutEndNode>(blackboard, name, testIndex, numTests, 30)));
            }

            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         std::move(disableAllBothDuringSetup),
                                                         make_unique<StartNode>(blackboard, name, testIndex, numTests),
                                                         std::move(finishCondition),
                                                         make_unique<SuccessEndNode>(blackboard, name, testIndex, numTests)),
                                                 "LearnedNavSequence");
        }
    }


    LearnedHoldASitePushScript::LearnedHoldASitePushScript(std::size_t testIndex, std::size_t numTests, bool waitForever) :
            LearnedHoldScript("LearnedHoldASitePushScript",
                              {{0, ENGINE_TEAM_T, AggressiveType::Push}, {0, ENGINE_TEAM_T, AggressiveType::Push},
                               {0, ENGINE_TEAM_T, AggressiveType::Push}},
                              {ObserveType::Absolute, 0, {395.317963, 2659.722656, 559.311157}, {43.801949, -49.044704}},
                              testIndex, numTests, waitForever) { }

    void LearnedHoldASitePushScript::initialize(Tree &tree, ServerState &state) {
        vector<Vec3> playerPos {
            {1071.936035, 2972.308837, 128.762023}, {824.582764, 2612.630127, 95.957748},
            {1461.081055, 2392.754639, 22.165134},
        };
        vector<Vec2> playerViewAngle {
            {2.903987, -95.587982}, {-1.760050, -105.049713}, {-89.683349, 0.746031}
        };
        vector<AreaId> chokeAreas {3653, 1384, 4051};
        LearnedHoldScript::initialize(tree, state, playerPos, playerViewAngle, Vec3{1241., 2586., 127.}, chokeAreas);
    }

    LearnedHoldASiteBaitScript::LearnedHoldASiteBaitScript(std::size_t testIndex, std::size_t numTests, bool waitForever) :
            LearnedHoldScript("LearnedHoldASiteBaitScript",
                              {{0, ENGINE_TEAM_T, AggressiveType::Bait}, {0, ENGINE_TEAM_T, AggressiveType::Bait},
                               {0, ENGINE_TEAM_T, AggressiveType::Bait}},
                              {ObserveType::Absolute, 0, {395.317963, 2659.722656, 559.311157}, {43.801949, -49.044704}},
                              testIndex, numTests, waitForever) { }

    void LearnedHoldASiteBaitScript::initialize(Tree &tree, ServerState &state) {
        vector<Vec3> playerPos {
            {1071.936035, 2972.308837, 128.762023}, {824.582764, 2612.630127, 95.957748},
            {1461.081055, 2392.754639, 22.165134},
        };
        vector<Vec2> playerViewAngle {
           {2.903987, -95.587982}, {-1.760050, -105.049713}, {-89.683349, 0.746031}
        };
        vector<AreaId> chokeAreas {4170, 9018, 4048};
        LearnedHoldScript::initialize(tree, state, playerPos, playerViewAngle, Vec3{1241., 2586., 127.}, chokeAreas);
    }

    LearnedHoldBSitePushScript::LearnedHoldBSitePushScript(std::size_t testIndex, std::size_t numTests, bool waitForever) :
            LearnedHoldScript("LearnedHoldBSitePushScript",
                              {{0, ENGINE_TEAM_T, AggressiveType::Push}, {0, ENGINE_TEAM_T, AggressiveType::Push},
                               {0, ENGINE_TEAM_T, AggressiveType::Push}},
                              {ObserveType::Absolute, 0, {-2092., 3050., 710.}, {56., -68.}},
                              testIndex, numTests, waitForever) { }

    void LearnedHoldBSitePushScript::initialize(Tree &tree, ServerState &state) {
        vector<Vec3> playerPos {
            {-1990., 2644., 93.6}, {-1438., 2461., 65.5}, Vec3{-1750., 1868., 64.2},
        };
        vector<Vec2> playerViewAngle {
            {2.903987, -95.587982}, {-1.760050, -105.049713}, {-89.683349, 0.746031}
        };
        vector<AreaId> chokeAreas {1896, 8489, 556};
        LearnedHoldScript::initialize(tree, state, playerPos, playerViewAngle, Vec3{-1574., 2638., 38.}, chokeAreas);
    }

    LearnedHoldBSiteBaitScript::LearnedHoldBSiteBaitScript(std::size_t testIndex, std::size_t numTests, bool waitForever) :
            LearnedHoldScript("LearnedHoldBSiteBaitScript",
                              {{0, ENGINE_TEAM_T, AggressiveType::Bait}, {0, ENGINE_TEAM_T, AggressiveType::Bait},
                               {0, ENGINE_TEAM_T, AggressiveType::Bait}},
                              {ObserveType::Absolute, 0, {-2092., 3050., 710.}, {56., -68.}},
                              testIndex, numTests, waitForever) { }

    void LearnedHoldBSiteBaitScript::initialize(Tree &tree, ServerState &state) {
        vector<Vec3> playerPos {
            {-1990., 2644., 93.6}, {-1438., 2461., 65.5}, Vec3{-1750., 1868., 64.2},
        };
        vector<Vec2> playerViewAngle {
            {2.903987, -95.587982}, {-1.760050, -105.049713}, {-89.683349, 0.746031}
        };
        vector<AreaId> chokeAreas {7533, 8666, 8212};
        LearnedHoldScript::initialize(tree, state, playerPos, playerViewAngle, Vec3{-1574., 2638., 38.}, chokeAreas);
    }

    vector<Script::Ptr> createLearnedHoldScripts(size_t numTests, bool quitAtEnd) {
        vector<Script::Ptr> result;

        for (size_t i = 0; i < numTests; i++) {
            //result.push_back(make_unique<LearnedHoldASitePushScript>(i, numTests, false));
            //result.push_back(make_unique<LearnedHoldASiteBaitScript>(i, numTests, false));
            //result.push_back(make_unique<LearnedHoldBSitePushScript>(i, numTests, false));
            result.push_back(make_unique<LearnedHoldBSiteBaitScript>(i, numTests, false));
        }
        if (quitAtEnd) {
            result.push_back(make_unique<QuitScript>());
        }

        return result;
    }

}
