//
// Created by steam on 6/30/22.
//

#ifndef CSKNOW_TEAMWORK_H
#define CSKNOW_TEAMWORK_H

#include "bots/testing/script.h"
#include "bots/behavior_tree/pathing_node.h"
#include "bots/behavior_tree/tree.h"
#include "bots/testing/blackboard_management.h"

class PusherReachesCatBeforeBaiter : public Node {
    CSGOId pusherId, baiterId;

public:
    PusherReachesCatBeforeBaiter(Blackboard & blackboard, CSGOId pusherId, CSGOId baiterId) :
            Node(blackboard, "ValidConditionNod"), pusherId(pusherId), baiterId(baiterId) { };
    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        const ServerState::Client & pusherClient = state.getClient(pusherId);
        uint32_t pusherArea = blackboard.navFile.get_nearest_area_by_position(vec3Conv(pusherClient.getFootPosForPlayer())).get_id();
        string pusherPlace = blackboard.navFile.get_place(blackboard.navFile.get_area_by_id_fast(pusherArea).m_place);

        const ServerState::Client & baiterClient = state.getClient(baiterId);
        uint32_t baiterArea = blackboard.navFile.get_nearest_area_by_position(vec3Conv(baiterClient.getFootPosForPlayer())).get_id();
        string baiterPlace = blackboard.navFile.get_place(blackboard.navFile.get_area_by_id_fast(baiterArea).m_place);

        if (pusherPlace == "Catwalk") {
            playerNodeState[treeThinker.csgoId] = baiterPlace == "ShortStairs" ? NodeState::Success : NodeState::Failure;
            return playerNodeState[treeThinker.csgoId];
        }
        else {
            playerNodeState[treeThinker.csgoId] = NodeState::Running;
            return NodeState::Running;
        }
    }
};

class PushBaitGooseToCatScript : public Script {
public:
    PushBaitGooseToCatScript(const ServerState & state) :
            Script("PushBaitGooseToLongScript", {{0, ENGINE_TEAM_T}, {0, ENGINE_TEAM_T}}, {ObserveType::FirstPerson, 0}) { }

    virtual void initialize(Tree & tree, ServerState & state) override  {
        if (tree.newBlackboard)  {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            vector<string> aToCatPathPlace(order::catToAPathPlace.rbegin(), order::catToAPathPlace.rend());
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         make_unique<InitTestingRound>(blackboard),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id, neededBots[1].id},state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({1071.936035, 2972.308837, 128.762023}), Vec2({2.903987, -95.587982})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard,neededBots[0].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({824.582764, 2612.630127, 158.957748}), Vec2({-1.760050, -105.049713})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard, neededBots[1].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<ForceOrderNode>(blackboard, "ForceTCat", vector{neededBots[0].id, neededBots[1].id}, aToCatPathPlace),
                                                         make_unique<ForceAggressionNode>(blackboard, "ForcePusherBaiter",
                                                                                          vector{neededBots[0].id, neededBots[1].id},
                                                                                          vector{AggressiveType::Push, AggressiveType::Bait}),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                                make_unique<PusherReachesCatBeforeBaiter>(blackboard, neededBots[0].id, neededBots[1].id),
                                                                                                make_unique<movement::WaitNode>(blackboard, 20, false)),
                                                                                        "PushBaitGooseToLongCondition")),
                                                 "PushBaitGooseToLongSequence");
        }
    }
};

#endif //CSKNOW_TEAMWORK_H
