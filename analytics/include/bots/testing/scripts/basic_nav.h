//
// Created by steam on 6/20/22.
//

#ifndef CSKNOW_BASIC_NAV_H
#define CSKNOW_BASIC_NAV_H

#include "bots/testing/script.h"
#include "bots/behavior_tree/pathing_node.h"
#include "bots/behavior_tree/tree.h"
#include "bots/testing/blackboard_management.h"

class JumpedBeforeCat : public Node {
    CSGOId targetId;
    bool reachedBoxed = false;
public:
    JumpedBeforeCat(Blackboard & blackboard, CSGOId targetId) :
            Node(blackboard, "ValidConditionNod"), targetId(targetId) { };
    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        const ServerState::Client & curClient = state.getClient(targetId);
        uint32_t curArea = blackboard.navFile.get_nearest_area_by_position(vec3Conv(curClient.getFootPosForPlayer())).get_id();
        string curPlace = blackboard.navFile.get_place(blackboard.navFile.get_area_by_id_fast(curArea).m_place);
        // fail if get to cat before jumping on the boxes
        /*
        if (curArea == 4048) {
            set<uint32_t> badSources = blackboard.navFile.get_sources_to_area(4048);
            std::cout << "what happened here folks?" << std::endl;
        }
        for (const auto & waypoint : blackboard.playerToPath[targetId].waypoints) {
            if (waypoint.area1 == 4048 || waypoint.area2 == 4048) {
                set<uint32_t> badSources = blackboard.navFile.get_sources_to_area(4048);
                std::cout << "why happened here folks?" << std::endl;
            }
        }
         */
        if (curPlace == "Catwalk") {
            playerNodeState[treeThinker.csgoId] = reachedBoxed ? NodeState::Success : NodeState::Failure;
            return playerNodeState[treeThinker.csgoId];
        }
        else {
            if (curArea == 8799) {
                reachedBoxed = true;
            }
            playerNodeState[treeThinker.csgoId] = NodeState::Running;
            return NodeState::Running;
        }
    }

    virtual void restart(const TreeThinker & treeThinker) override {
        Node::restart(treeThinker);
        reachedBoxed = false;
    }
};

class GooseToCatScript : public Script {
public:
    GooseToCatScript(const ServerState & state) :
        Script("GooseToLongScript", {{0, ENGINE_TEAM_T}}, {ObserveType::FirstPerson, 0}) { }

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
                                                         make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id},state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({1071.936035, 2972.308837, 128.762023}), Vec2({2.903987, -95.587982})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard, neededBots[0].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<ForceOrderNode>(blackboard, "ForceTCat", vector{neededBots[0].id}, aToCatPathPlace),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                                make_unique<JumpedBeforeCat>(blackboard, neededBots[0].id),
                                                                                                make_unique<movement::WaitNode>(blackboard, 20, false)),
                                                                                        "GooseToLongCondition")),
                                                 "GooseToLongSequence");
        }
    }
};

#endif //CSKNOW_BASIC_NAV_H
