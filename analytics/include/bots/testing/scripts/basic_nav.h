//
// Created by steam on 6/20/22.
//

#ifndef CSKNOW_BASIC_NAV_H
#define CSKNOW_BASIC_NAV_H

#include "bots/testing/script.h"

class ForceTCatOrderNode : public Node {
    CSGOId targetId;
public:
    ForceTCatOrderNode(Blackboard & blackboard, CSGOId targetId) :
        Node(blackboard, "ForceTCatOrderNode"), targetId(targetId) { };
    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        blackboard.orders[3].followers.clear();
        blackboard.orders[4].followers.clear();
        blackboard.orders[5].followers = {targetId};
        blackboard.playerToOrder[targetId] = 5;
        return NodeState::Success;
    }
};

class JumpedBeforeCat : public Node {
    CSGOId targetId;
public:
    JumpedBeforeCat(Blackboard & blackboard, CSGOId targetId) :
            Node(blackboard, "ValidConditionNod"), targetId(targetId) { };
    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        const ServerState::Client & curClient = state.getClient(targetId);
        uint32_t curArea = blackboard.navFile.get_nearest_area_by_position(vec3Conv(curClient.getFootPosForPlayer())).get_id();
        string curPlace = blackboard.navFile.get_place(blackboard.navFile.get_area_by_id_fast(curArea).m_place);
        // fail if get to cat before jumping on the boxes
        if (curPlace == "Catwalk") {
            return NodeState::Failure;
        }
        else if (curArea == 8799) {
            return NodeState::Success;
        }
        else {
            return NodeState::Running;
        }
    }
};

class GooseToCatScript : public Script {
public:
    GooseToCatScript(Blackboard & blackboard) :
        Script(blackboard, "GooseToLongScript", {{0, ENGINE_TEAM_T}}, {ObserveType::FirstPerson, 0}) { }

    virtual vector<string> generateCommands(ServerState & state) override {
        logicCommands = Command::makeList(
                make_unique<SetPos>(Vec3({1071.936035, 2972.308837, 128.762023}), Vec2({2.903987, -95.587982})),
                make_unique<Teleport>(neededBots[0].id, state));

        conditions = make_unique<SequenceNode>(blackboard, Node::makeList(
                    make_unique<ForceTCatOrderNode>(blackboard, neededBots[0].id),
                    make_unique<JumpedBeforeCat>(blackboard, neededBots[0].id)
                ), "GooseToLongRootNode");

        return Script::generateCommands(state);
    }
};

#endif //CSKNOW_BASIC_NAV_H
