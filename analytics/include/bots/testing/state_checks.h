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
};

class AimingAtArea : public Node {
    vector<CSGOId> sourceIds;
    AreaId targetId;
    bool invert, requireVisible;

public:
    AimingAtArea(Blackboard & blackboard, vector<CSGOId> sourceIds, AreaId targetId, bool invert = false, bool requireVisible = true) :
            Node(blackboard, "AimingAtNode"), sourceIds(sourceIds), targetId(targetId), invert(invert), requireVisible(requireVisible) { };

    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        bool aimingAtTarget = false;

        for (const auto & sourceId : sourceIds) {
            const ServerState::Client & sourceClient = state.getClient(sourceId);

            AreaId srcAreaId =
                    blackboard.navFile.get_nearest_area_by_position(vec3Conv(sourceClient.getFootPosForPlayer())).get_id();
            if (requireVisible && !blackboard.visPoints.isVisibleAreaId(srcAreaId, targetId))  {
                continue;
            }

            if (targetId == 3777) {
                int x = 1;
            }

            // check if aiming at enemy anywhere
            Ray eyeCoordinates = getEyeCoordinatesForPlayerGivenEyeHeight(
                    sourceClient.getEyePosForPlayer(),
                    sourceClient.getCurrentViewAnglesWithAimpunch());

            AABB targetAABB(areaToAABB(blackboard.navFile.get_area_by_id_fast(targetId)));
            targetAABB.max.z += 5 * EYE_HEIGHT; // a lot of leeway in z direction, don't really care here
            double hitt0, hitt1;
            aimingAtTarget |= intersectP(targetAABB, eyeCoordinates, hitt0, hitt1);
        }

        if ((!invert && aimingAtTarget) || (invert && !aimingAtTarget)) {
            playerNodeState[treeThinker.csgoId] = NodeState::Success;
        }
        else {
            playerNodeState[treeThinker.csgoId] = NodeState::Failure;
        }
        return playerNodeState[treeThinker.csgoId];
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
};

enum class PosConstraintDimension {
    X,
    Y,
    Z
};

enum class PosConstraintOp {
   LT,
   LTE,
   GT,
   GTE,
   EQ
};

string posConstraintOpToString(const PosConstraintOp & op) {
    switch (op) {
        case PosConstraintOp::LT:
            return "<";
        case PosConstraintOp::LTE:
            return "<=";
        case PosConstraintOp::GT:
            return ">";
        case PosConstraintOp::GTE:
            return ">=";
        case PosConstraintOp::EQ:
            return "==";
        default:
            throw std::runtime_error("invalid pos constraint in to string");
    }

}

class PosConstraint : public Node {
    CSGOId playerId;
    PosConstraintDimension dim;
    PosConstraintOp op;
    double value;

public:
    PosConstraint(Blackboard & blackboard, CSGOId playerId, PosConstraintDimension dim, PosConstraintOp op, double value) :
            Node(blackboard, "PosConstraint" + posConstraintOpToString(op) + std::to_string(value)), playerId(playerId),
            dim(dim), op(op), value(value) { };

    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        const ServerState::Client & sourceClient = state.getClient(playerId);

        double testValue;
        switch (dim) {
            case PosConstraintDimension::X:
                testValue = sourceClient.getFootPosForPlayer().x;
                break;
            case PosConstraintDimension::Y:
                testValue = sourceClient.getFootPosForPlayer().y;
                break;
            case PosConstraintDimension::Z:
                testValue = sourceClient.getFootPosForPlayer().z;
                break;
            default:
                throw std::runtime_error("invalid dim in pos constraint");
        }

        switch (op) {
            case PosConstraintOp::LT:
                playerNodeState[treeThinker.csgoId] = testValue < value ? NodeState::Success : NodeState::Failure;
                break;
            case PosConstraintOp::LTE:
                playerNodeState[treeThinker.csgoId] = testValue <= value ? NodeState::Success : NodeState::Failure;
                break;
            case PosConstraintOp::GT:
                playerNodeState[treeThinker.csgoId] = testValue > value ? NodeState::Success : NodeState::Failure;
                break;
            case PosConstraintOp::GTE:
                playerNodeState[treeThinker.csgoId] = testValue >= value ? NodeState::Success : NodeState::Failure;
                break;
            case PosConstraintOp::EQ:
                playerNodeState[treeThinker.csgoId] = testValue == value ? NodeState::Success : NodeState::Failure;
                break;
            default:
                throw std::runtime_error("invalid op in pos constraint");

        }

        return playerNodeState[treeThinker.csgoId];
    }
};

#endif //CSKNOW_STATE_CHECKS_H
