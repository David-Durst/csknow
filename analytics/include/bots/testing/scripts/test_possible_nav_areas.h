//
// Created by steam on 7/15/22.
//

#ifndef CSKNOW_TEST_POSSIBLE_NAV_AREAS_H
#define CSKNOW_TEST_POSSIBLE_NAV_AREAS_H

#include "bots/testing/script.h"
#include "bots/behavior_tree/pathing_node.h"
#include "bots/behavior_tree/tree.h"
#include "bots/testing/blackboard_management.h"
#include "bots/testing/state_checks.h"

class SpawnPossibleNavAreasCheck : public Script {
public:
    SpawnPossibleNavAreasCheck(const ServerState & state) :
            Script("SpawnPossibleNavAreasCheck", {{0, ENGINE_TEAM_CT}, {0, ENGINE_TEAM_CT}, {0, ENGINE_TEAM_T}, {0, ENGINE_TEAM_T}},
                   {ObserveType::Absolute, 0, {395.317963, 2659.722656, 559.311157}, {43.801949, -49.044704}}) { }

    virtual void initialize(Tree & tree, ServerState & state) override {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            Vec3 player0Pos{1417.528564, 1652.856445, -9.775691}, player1Pos{1461.081055, 2392.754639, 22.165134};
            vector<AreaId> onePlayerRequiredPossibleAreas{blackboard.navFile.get_nearest_area_by_position(vec3Conv(player0Pos)).get_id(),
                                                          blackboard.navFile.get_nearest_area_by_position(vec3Conv(player1Pos)).get_id()},
                                                          onePlayerRequiredNotPossibleAreas{3734, 8092, 6992};
            Node::Ptr setupCommands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                                        make_unique<InitTestingRound>(blackboard, name),
                                                                        make_unique<movement::WaitNode>(blackboard, 1.0),
                                                                        make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id, neededBots[1].id}, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.3),
                                                                        make_unique<SetPos>(blackboard, player0Pos, Vec2({-89.683349, 0.746031})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[0].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SetPos>(blackboard, player1Pos, Vec2({-89.683349, 0.746031})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[1].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 2.0),
                                                                        make_unique<ClearMemoryCommunicationDangerNode>(blackboard),
                                                                        make_unique<movement::WaitTicksNode>(blackboard, 1)));
            Node::Ptr disableAllBothDuringSetup = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                    std::move(setupCommands),
                    make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[0].id, neededBots[1].id, neededBots[2].id, neededBots[3].id}, false)
            ));
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         std::move(disableAllBothDuringSetup),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                                make_unique<DisableActionsNode>(blackboard, "DisableCondition", vector{neededBots[0].id, neededBots[1].id}),
                                                                                                make_unique<CheckPossibleLocationsNode>(blackboard, vector{neededBots[0].id, neededBots[1].id},
                                                                                                                                        vector{onePlayerRequiredPossibleAreas, onePlayerRequiredPossibleAreas},
                                                                                                                                        vector{onePlayerRequiredNotPossibleAreas, onePlayerRequiredNotPossibleAreas}))
                                                                 //make_unique<movement::WaitNode>(blackboard, 0.8)),
                                                                                        )));
        }
    }
};

#endif //CSKNOW_TEST_POSSIBLE_NAV_AREAS_H
