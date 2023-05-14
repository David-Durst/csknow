//
// Created by steam on 6/20/22.
//

#ifndef CSKNOW_ENTRY_H
#define CSKNOW_ENTRY_H

#include <utility>

#include "bots/testing/script.h"
#include "bots/behavior_tree/pathing_node.h"
#include "bots/behavior_tree/tree.h"
#include "bots/testing/blackboard_management.h"
#include "bots/testing/state_checks.h"
#include "bots/analysis/learned_models.h"

class JumpedBeforeCat : public Node {
    CSGOId targetId;
    bool reachedBoxed = false;
public:
    JumpedBeforeCat(Blackboard & blackboard, CSGOId targetId) :
            Node(blackboard, "ValidConditionNode"), targetId(targetId) { };
    NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        const ServerState::Client & curClient = state.getClient(targetId);
        uint32_t curArea = blackboard.navFile.get_nearest_area_by_position(vec3Conv(curClient.getFootPosForPlayer())).get_id();
        string curPlace = blackboard.navFile.get_place(blackboard.navFile.get_area_by_id_fast(curArea).m_place);
        // fail if get to cat before jumping on the boxes
        /*
        if (curArea == 4048) {
            set<uint32_t> badSources = blackboard.navFile.get_sources_to_area(4048);
            std::cout << "what happened here folks?" << std::endl;
        }
        for (const auto & waypoint : blackboard.playerToPath[csgoId].waypoints) {
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

    void restart(const TreeThinker & treeThinker) override {
        Node::restart(treeThinker);
        reachedBoxed = false;
    }
};

class GooseToCatScript : public Script {
public:
    OrderId addedOrderId;

    explicit GooseToCatScript(const ServerState &) :
            Script("GooseToCatScript", {{0, ENGINE_TEAM_CT}}, {ObserveType::FirstPerson, 0}) { }

    void initialize(Tree & tree, ServerState & state) override  {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            set<AreaId> areasToRemove{};
            if (!getPlaceAreaModelProbabilities(ENGINE_TEAM_CT)) {
                areasToRemove.insert(4048);
            }
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         make_unique<InitTestingRound>(blackboard, name),
                                                         make_unique<movement::WaitNode>(blackboard, 1.0),
                                                         make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id},state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({1071.936035, 2972.308837, 128.762023}), Vec2({-84.903987, -95.587982})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard, neededBots[0].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 1.0),
                                                         make_unique<ForceOrderNode>(blackboard, "ForceTCat", vector{neededBots[0].id}, testAToCatWaypoints, areasToRemove, addedOrderId),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                                make_unique<JumpedBeforeCat>(blackboard, neededBots[0].id),
                                                                                                make_unique<movement::WaitNode>(blackboard, 20, false)),
                                                                                        "GooseToLongCondition")),
                                                 "GooseToLongSequence");
        }
    }
};

class DontEnterNavAreas : public Node {
    CSGOId targetId;
    set<uint32_t> forbiddenAreas;
public:
    DontEnterNavAreas(Blackboard & blackboard, CSGOId targetId, set<uint32_t> forbiddenAreas) :
            Node(blackboard, "DontEnterNavAreas"), targetId(targetId), forbiddenAreas(std::move(forbiddenAreas)) { };
    NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        const ServerState::Client & curClient = state.getClient(targetId);
        uint32_t curArea = blackboard.navFile.get_nearest_area_by_position(vec3Conv(curClient.getFootPosForPlayer())).get_id();
        if (forbiddenAreas.find(curArea) == forbiddenAreas.end()) {
            playerNodeState[treeThinker.csgoId] = NodeState::Running;
        }
        else {
            playerNodeState[treeThinker.csgoId] = NodeState::Failure;
        }
        return playerNodeState[treeThinker.csgoId];
    }
};

class GooseToCatShortScript : public Script {
public:
    OrderId addedOrderId;

    explicit GooseToCatShortScript(const ServerState &) :
            Script("GooseToCatShortScript", {{0, ENGINE_TEAM_CT}}, {ObserveType::FirstPerson, 0}) { }

    void initialize(Tree & tree, ServerState & state) override  {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            set<AreaId> areasToRemove{};
            if (!getPlaceAreaModelProbabilities(ENGINE_TEAM_CT)) {
                areasToRemove.insert(4048);
            }
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         make_unique<InitTestingRound>(blackboard, name),
                                                         make_unique<movement::WaitNode>(blackboard, 1.0),
                                                         make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id},state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({420.199219, 2277.000000, 159.528168}), Vec2({-0.659997, 5.090078})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard, neededBots[0].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 1.0),
                                                         make_unique<ForceOrderNode>(blackboard, "ForceTCat", vector{neededBots[0].id}, testAToCatWaypoints, areasToRemove, addedOrderId),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                                make_unique<DontEnterNavAreas>(blackboard, neededBots[0].id, set{1722u, 1723u, 1727u}),
                                                                                                make_unique<JumpedBeforeCat>(blackboard, neededBots[0].id),
                                                                                                make_unique<movement::WaitNode>(blackboard, 20, false)),
                                                                                        "GooseToLongCondition")),
                                                 "GooseToLongSequence");
        }
    }
};

class CTPushLongScript : public Script {
public:
    OrderId addedOrderId;

    explicit CTPushLongScript(const ServerState &) :
            Script("CTPushLongScript", {{0, ENGINE_TEAM_CT}}, {ObserveType::FirstPerson, 0}) { }

    void initialize(Tree & tree, ServerState & state) override  {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            set<AreaId> areasToRemove{4048};
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         make_unique<InitTestingRound>(blackboard, name),
                                                         make_unique<movement::WaitNode>(blackboard, 1.0),
                                                         make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id},state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({593., 282., 2.}), Vec2({2.903987, -95.587982})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard, neededBots[0].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<ForceOrderNode>(blackboard, "ForceCTLong", vector{neededBots[0].id}, strategy::offenseLongToAWaypoints, areasToRemove, addedOrderId),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                 make_unique<RepeatDecorator>(blackboard, make_unique<InPlace>(blackboard, neededBots[0].id, "BombsiteA"), true),
                                                                 make_unique<movement::WaitNode>(blackboard, 20, false))
                                                                 ))
                                                );
        }
    }
};

class CTPushBDoorsScript : public Script {
public:
    OrderId addedOrderId;

    explicit CTPushBDoorsScript(const ServerState &) :
            Script("CTPushBDoorsScript", {{0, ENGINE_TEAM_CT}}, {ObserveType::FirstPerson, 0}) { }

    void initialize(Tree & tree, ServerState & state) override  {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            set<AreaId> areasToRemove{4048};
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                    make_unique<InitTestingRound>(blackboard, name),
                    make_unique<movement::WaitNode>(blackboard, 1.0),
                    make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id},state),
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    make_unique<SetPos>(blackboard, Vec3({-516., 1733., -14.}), Vec2({-89.683349, 0.746031})),
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    make_unique<Teleport>(blackboard, neededBots[0].id, state),
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    make_unique<ForceOrderNode>(blackboard, "ForceBDoorsLong", vector{neededBots[0].id}, strategy::offenseBDoorsToBWaypoints, areasToRemove, addedOrderId),
                    make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                            make_unique<RepeatDecorator>(blackboard, make_unique<InPlace>(blackboard, neededBots[0].id, "BombsiteB"), true),
                            make_unique<movement::WaitNode>(blackboard, 20, false))
                    ))
            );
        }
    }
};

class CTPushBHoleScript : public Script {
public:
    OrderId addedOrderId;

    explicit CTPushBHoleScript(const ServerState &) :
            Script("CTPushBHoleScript", {{0, ENGINE_TEAM_CT}}, {ObserveType::FirstPerson, 0}) { }

    void initialize(Tree & tree, ServerState & state) override  {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            set<AreaId> areasToRemove{4048};
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                    make_unique<InitTestingRound>(blackboard, name),
                    make_unique<movement::WaitNode>(blackboard, 1.0),
                    make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id},state),
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    make_unique<SetPos>(blackboard, Vec3({-516., 1733., -14.}), Vec2({-89.683349, 0.746031})),
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    make_unique<Teleport>(blackboard, neededBots[0].id, state),
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    make_unique<ForceOrderNode>(blackboard, "ForceBHoleLong", vector{neededBots[0].id}, strategy::offenseHoleToBWaypoints, areasToRemove, addedOrderId),
                    make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                            make_unique<RepeatDecorator>(blackboard, make_unique<InPlace>(blackboard, neededBots[0].id, "BombsiteB"), true),
                            make_unique<movement::WaitNode>(blackboard, 20, false))
                    ))
            );
        }
    }
};

#endif //CSKNOW_ENTRY_H
