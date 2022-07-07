//
// Created by steam on 7/6/22.
//

#ifndef CSKNOW_STATE_CHECKS_H
#define CSKNOW_STATE_CHECKS_H

#include "bots/testing/script.h"
#include "bots/behavior_tree/pathing_node.h"
#include "bots/behavior_tree/tree.h"
#include "bots/testing/blackboard_management.h"

class AimingAt : public Node {
    CSGOId sourceId, targetId;
    bool invert;

public:
    AimingAt(Blackboard & blackboard, CSGOId sourceId, CSGOId targetId, bool invert = false) :
            Node(blackboard, "AimingAtNode"), sourceId(sourceId), targetId(targetId), invert(invert) { };

    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        const ServerState::Client & sourceClient = state.getClient(sourceId);
        const ServerState::Client & targetClient = state.getClient(targetId);

        // check if aiming at enemy anywhere
        Ray eyeCoordinates = getEyeCoordinatesForPlayerGivenEyeHeight(
                sourceClient.getEyePosForPlayer(),
                sourceClient.getCurrentViewAnglesWithAimpunch());

        AABB targetAABB = getAABBForPlayer(targetClient.getFootPosForPlayer());
        double hitt0, hitt1;
        bool aimingAtEnemy = intersectP(targetAABB, eyeCoordinates, hitt0, hitt1);

        if ((!invert && aimingAtEnemy) || (invert && !aimingAtEnemy)) {
            playerNodeState[treeThinker.csgoId] = NodeState::Running;
        }
        else {
            playerNodeState[treeThinker.csgoId] = NodeState::Failure;
        }
        return playerNodeState[treeThinker.csgoId];
    }

    virtual void restart(const TreeThinker & treeThinker) override {
        Node::restart(treeThinker);
    }
};

class Firing : public Node {
    CSGOId sourceId;
    bool invert;

public:
    Firing(Blackboard & blackboard, CSGOId sourceId, bool invert = false) :
            Node(blackboard, "FiringNode"), sourceId(sourceId), invert(invert) { };

    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        const ServerState::Client & sourceClient = state.getClient(sourceId);

        if ((!invert && blackboard.playerToAction[sourceId].getButton(IN_ATTACK)) ||
            (invert && !blackboard.playerToAction[sourceId].getButton(IN_ATTACK))) {
            playerNodeState[treeThinker.csgoId] = NodeState::Running;
        }
        else {
            playerNodeState[treeThinker.csgoId] = NodeState::Failure;
        }
        return playerNodeState[treeThinker.csgoId];
    }

    virtual void restart(const TreeThinker & treeThinker) override {
        Node::restart(treeThinker);
    }
};

#endif //CSKNOW_STATE_CHECKS_H
