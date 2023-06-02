//
// Created by durst on 5/19/23.
//

#include "bots/testing/scripts/learned/test_learned_teamwork.h"
#include "bots/testing/scripts/learned/log_nodes.h"
#include "bots/testing/scripts/test_teamwork.h"

namespace csknow::tests::learned {
    LearnedTeamworkScript::LearnedTeamworkScript(
            const std::string &name, vector<NeededBot> neededBots, ObserveSettings observeSettings,
            std::size_t testIndex, std::size_t numTests, bool waitForever) :
            Script(name, neededBots, observeSettings),
            testIndex(testIndex), numTests(numTests), waitForever(waitForever) { }

    void LearnedTeamworkScript::initialize(Tree &tree, ServerState &state, vector<Vec3> playerPos,
                                           vector<Vec2> playerViewAngle, Vec3 c4Pos,
                                           Node::Ptr forceSetup, Node::Ptr condition) {
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
                                                                        make_unique<ClearMemoryCommunicationDangerNode>(blackboard),
                                                                        std::move(forceSetup),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<RepeatDecorator>(blackboard,
                                                                                                     make_unique<StandingStill>(blackboard, neededBotIds), true)),
                                                                "Setup");

            Node::Ptr disableAllBothDuringSetup = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                    std::move(setupCommands),
                    make_unique<DisableActionsNode>(blackboard, "DisableSetup", neededBotIds, false)
            ), "DisableDuringSetup");

            Node::Ptr finishCondition;
            if (waitForever) {
                finishCondition = make_unique<RepeatDecorator>(blackboard, std::move(condition), true);
            }
            else {
                finishCondition = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                        make_unique<RepeatDecorator>(blackboard, std::move(condition), true),
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

    LearnedPushLurkBaitASiteScript::LearnedPushLurkBaitASiteScript(std::size_t testIndex, std::size_t numTests, bool waitForever) :
            LearnedTeamworkScript("LearnedPushLurkBaitASiteScript",
                                  {{0, ENGINE_TEAM_CT, AggressiveType::Bait}, {0, ENGINE_TEAM_CT, AggressiveType::Bait},
                                   {0, ENGINE_TEAM_CT, AggressiveType::Push}/*, {0, ENGINE_TEAM_T}*/},
                                   {ObserveType::FirstPerson, 0},
                                   testIndex, numTests, waitForever) { }

    void LearnedPushLurkBaitASiteScript::initialize(Tree &tree, ServerState &state) {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            vector<Vec3> playerPos {
                    {473.290436, -67.194908, 59.092133}, {-14.934761, -817.601318, 62.097897},
                    {-421.860260, 856.695313, 42.407509}//, {883.084106, 2491.471436, 160.187653}
            };
            vector<Vec2> playerViewAngle {
                    {-78.701942, -7.463999}, {-89.683349, 0.746031}, {-89.683349, 0.746031}//, {-89.683349, 0.746031}
            };

            Node::Ptr forceSetup = make_unique<SequenceNode>(blackboard, Node::makeList(
                    make_unique<ForceOrderNode>(blackboard, "ForceCTLongCat", vector{neededBots[0].id, neededBots[1].id}, strategy::offenseLongToAWaypoints, lurkAddedOrderId),
                    make_unique<ForceOrderNode>(blackboard, "ForceCTCat", vector{neededBots[2].id}, strategy::offenseCatToAWaypoints, pushAddedOrderId),
                    make_unique<ForceEntryIndexNode>(blackboard, "ForcePusherBaiter",
                                                     vector{neededBots[0].id, neededBots[1].id, neededBots[2].id},
                                                     vector{0, 1, 0}),
                    make_unique<movement::WaitNode>(blackboard, 2.0)), "ForceSetup");

            Node::Ptr pusherSeesEnemyBeforeLurkerMoves =
                    make_unique<RepeatDecorator>(blackboard, make_unique<ParallelAndNode>(blackboard, Node::makeList(
                            make_unique<InPlace>(blackboard, neededBots[0].id, "LongDoors"),
                            make_unique<PosConstraint>(blackboard, neededBots[0].id, PosConstraintDimension::Y, PosConstraintOp::LT, 450),
                            make_unique<InPlace>(blackboard, neededBots[1].id, "OutsideLong"),
                            make_unique<InPlace>(blackboard, neededBots[2].id, "ExtendedA")
            )), true);

            Node::Ptr pusherHoldsLurkerReachesLongBaiterFollows =
                    make_unique<RepeatDecorator>(blackboard, make_unique<ParallelAndNode>(blackboard, Node::makeList(
                            make_unique<InPlace>(blackboard, neededBots[0].id, "LongA"),
                            make_unique<InPlace>(blackboard, neededBots[1].id, "LongA"),
                            make_unique<InPlace>(blackboard, neededBots[2].id, "ExtendedA")
            )), true);
            Node::Ptr placeChecks = make_unique<SequenceNode>(blackboard, Node::makeList(
                    std::move(pusherSeesEnemyBeforeLurkerMoves),
                    std::move(pusherHoldsLurkerReachesLongBaiterFollows)
            ));
            Node::Ptr condition = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                    //make_unique<DisableActionsNode>(blackboard, "DisableEnemy", vector{neededBots[3].id}),
                    make_unique<DisableActionsNode>(blackboard, "DisablePush", vector{neededBots[2].id}, false, true, false),
                    std::move(placeChecks)),
                "Condition");

            LearnedTeamworkScript::initialize(tree, state, playerPos, playerViewAngle, Vec3{1241., 2586., 127.},
                                              std::move(forceSetup), std::move(condition));
        }
    }

    LearnedPushATwoOrdersScript::LearnedPushATwoOrdersScript(std::size_t testIndex, std::size_t numTests, bool waitForever) :
            LearnedTeamworkScript("LearnedPushATwoOrdersScript",
                                  {{0, ENGINE_TEAM_CT}, {0, ENGINE_TEAM_CT}},
                                  {ObserveType::FirstPerson, 1},
                                  testIndex, numTests, waitForever) { }

    void LearnedPushATwoOrdersScript::initialize(Tree &tree, ServerState &state) {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            vector<Vec3> playerPos {
                    {-376., 729., 64.}, {-944., 1440., -49.}
            };
            vector<Vec2> playerViewAngle {
                    {2.903987, -95.587982}, {-1.760050, -105.049713}
            };

            Node::Ptr forceSetup = make_unique<SequenceNode>(blackboard, Node::makeList(
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    make_unique<RecomputeOrdersNode>(blackboard)), "ForceSetup");

            set<string> baiterValidLocations{"ShortStairs", "ExtendedA"};
            Node::Ptr condition = make_unique<PusherReachesBeforeBaiter>(blackboard, neededBots[1].id, neededBots[0].id, "UnderA", baiterValidLocations);

            LearnedTeamworkScript::initialize(tree, state, playerPos, playerViewAngle, Vec3{1241., 2586., 127.},
                                              std::move(forceSetup), std::move(condition));
        }
    }

    LearnedPushThreeBScript::LearnedPushThreeBScript(std::size_t testIndex, std::size_t numTests, bool waitForever) :
            LearnedTeamworkScript("LearnedPushThreeBScript",
                                  {{0, ENGINE_TEAM_CT}, {0, ENGINE_TEAM_CT}, {0, ENGINE_TEAM_CT}},
                                  {ObserveType::FirstPerson, 0},
                                  testIndex, numTests, waitForever) { }

    void LearnedPushThreeBScript::initialize(Tree &tree, ServerState &state) {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            vector<Vec3> playerPos {
                    {-224., 1225., 66.}, {344., 2292., -118.}, {-1591., 200., 129.}
            };
            vector<Vec2> playerViewAngle {
                    {2.903987, -95.587982}, {-1.760050, -105.049713}, {-89.683349, 0.746031}
            };

            Node::Ptr forceSetup = make_unique<SequenceNode>(blackboard, Node::makeList(
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    make_unique<RecomputeOrdersNode>(blackboard)), "ForceSetup");

            set<string> baiterValidLocations{"ShortStairs", "ExtendedA"};
            Node::Ptr condition =
                    make_unique<SequenceNode>(blackboard, Node::makeList(
                            make_unique<RepeatDecorator>(blackboard, make_unique<InPlace>(blackboard, neededBots[0].id, "BombsiteB"), true),
                            make_unique<RepeatDecorator>(blackboard, make_unique<InPlace>(blackboard, neededBots[1].id, "BDoors"), true),
                            make_unique<RepeatDecorator>(blackboard, make_unique<InPlace>(blackboard, neededBots[2].id, "UpperTunnel"), true)),
                "Condition");

            LearnedTeamworkScript::initialize(tree, state, playerPos, playerViewAngle, {-1463., 2489., 46.},
                                              std::move(forceSetup), std::move(condition));
        }
    }

    vector<Script::Ptr> createLearnedTeamworkScripts(size_t numTests, bool quitAtEnd) {
        vector<Script::Ptr> result;

        for (size_t i = 0; i < numTests; i++) {
            result.push_back(make_unique<LearnedPushLurkBaitASiteScript>(i, numTests, false));
            //result.push_back(make_unique<LearnedPushATwoOrdersScript>(i, numTests, false));
            //result.push_back(make_unique<LearnedPushThreeBScript>(i, numTests, false));
        }
        if (quitAtEnd) {
            result.push_back(make_unique<QuitScript>());
        }

        return result;
    }

}