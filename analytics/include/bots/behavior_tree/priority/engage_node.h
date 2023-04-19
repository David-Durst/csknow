//
// Created by durst on 6/9/22.
//

#ifndef CSKNOW_ENGAGE_NODE_H
#define CSKNOW_ENGAGE_NODE_H
#include "bots/behavior_tree/node.h"
#include "bots/behavior_tree/pathing_node.h"
#include <map>
#define COMMUNICATED_ENEMY_RELEVANT_TIME 2.0
#define MIN_ENGAGE_FRIENDLY_DISTANCE 100.

namespace engage {
    class RecordEngagementRound : public Node {
    public:
        RecordEngagementRound(Blackboard & blackboard) : Node(blackboard, "RecordEngagement") { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    class SelectTargetNode : public Node {
    public:
        SelectTargetNode(Blackboard & blackboard) : Node(blackboard, "SelectTargetNode") { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
        CSGOId assignPlayerToTargetProbabilistic(const ServerState::Client & client, const ServerState & state,
                                                 TargetPlayer & curTarget, CSGOId lastProbTargetAssignment,
                                                 const map<CSGOId, EnemyPositionMemory> & rememberedEnemies,
                                                 const map<CSGOId, EnemyPositionMemory> & communicatedEnemies);
    };

    class SelectFireModeNode : public Node {
    public:
        SelectFireModeNode(Blackboard & blackboard) : Node(blackboard, "FireSelectionTaskNode") { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    class SelectTargetAggressionNode : public Node {
    public:
        SelectTargetAggressionNode(Blackboard & blackboard) : Node(blackboard, "TargetAggressionSelectionTaskNode") { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };
}

class EngageNode : public SequenceNode {
public:
    EngageNode(Blackboard & blackboard) :
            SequenceNode(blackboard, Node::makeList(
                                 make_unique<engage::RecordEngagementRound>(blackboard),
                                 make_unique<engage::SelectTargetNode>(blackboard),
                                 make_unique<engage::SelectFireModeNode>(blackboard),
                                 make_unique<engage::SelectTargetAggressionNode>(blackboard),
                                 make_unique<movement::PathingNode>(blackboard)),
                         "EngageNode") { };
};

class EnemyEngageCheckNode : public ConditionDecorator {
public:
    EnemyEngageCheckNode(Blackboard & blackboard) : ConditionDecorator(blackboard,
                                                                        make_unique<EngageNode>(blackboard),
                                                                        "EnemyEngageCheckNode") { };

    void setPossibleEnemyDistances(Vec3 attackerEyePos, Vec2 attackerViewAngle,
                                   Vec3 victimEyePos, Vec2 victimViewAngle, double victimDuckAmount,
                                   csknow::feature_store::EngagementPossibleEnemy & engagementPossibleEnemy) {
        Vec3 victimHeadPos = getCenterHeadCoordinatesForPlayer(victimEyePos, victimViewAngle, victimDuckAmount);
        Vec2 deltaViewAngle = deltaViewFromOriginToDest(attackerEyePos, victimHeadPos, attackerViewAngle);
        engagementPossibleEnemy.worldDistanceToEnemy = computeDistance(attackerEyePos, victimEyePos);
        engagementPossibleEnemy.crosshairDistanceToEnemyHead = computeMagnitude(deltaViewAngle);
    }

    void setTeammateDistances(Vec3 attackerEyePos, Vec2 attackerViewAngle,
                                   Vec3 victimEyePos, Vec2 victimViewAngle, double victimDuckAmount,
                                   csknow::feature_store::EngagementTeammate & engagementTeammate) {
        Vec3 victimHeadPos = getCenterHeadCoordinatesForPlayer(victimEyePos, victimViewAngle, victimDuckAmount);
        Vec2 deltaViewAngle = deltaViewFromOriginToDest(attackerEyePos, victimHeadPos, attackerViewAngle);
        engagementTeammate.worldDistanceToTeammate = computeDistance(attackerEyePos, victimEyePos);
        engagementTeammate.crosshairDistanceToTeammateHead = computeMagnitude(deltaViewAngle);
    }

    virtual bool valid(const ServerState & state, TreeThinker & treeThinker) override {
        const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);
        bool enemyVisible = !state.getVisibleEnemies(treeThinker.csgoId).empty();
        bool rememberEnemy = !blackboard.playerToMemory[treeThinker.csgoId].positions.empty();

        map<CSGOId, csknow::feature_store::EngagementPossibleEnemy> possibleEnemies;
        for (const auto & visibleEnemy : state.getVisibleEnemies(treeThinker.csgoId)) {
            possibleEnemies[visibleEnemy.get().csgoId] = {
                visibleEnemy.get().csgoId, csknow::feature_store::EngagementEnemyState::Visible, 0.,
                INVALID_ID, INVALID_ID
            };
        }

        for (const auto & [rememberedEnemyId, rememberedEnemyState] :
             blackboard.playerToMemory[treeThinker.csgoId].positions) {
            if (possibleEnemies.find(rememberedEnemyId) == possibleEnemies.end()) {
                double secondsSinceLastSeen =
                    state.getSecondsBetweenTimes(state.loadTime, rememberedEnemyState.lastSeenTime);
                possibleEnemies[rememberedEnemyId] = {
                    rememberedEnemyId, csknow::feature_store::EngagementEnemyState::Remembered, secondsSinceLastSeen,
                    INVALID_ID, INVALID_ID
                };
            }
        }

        // select only relevant communicated enemies - those that are near a danger area
        map<CSGOId, EnemyPositionMemory> & relevantCommunicatedEnemies = blackboard.playerToRelevantCommunicatedEnemies[treeThinker.csgoId];
        relevantCommunicatedEnemies.clear();
        Vec3 curPos = state.getClient(treeThinker.csgoId).getFootPosForPlayer();
        AreaId curAreaId = blackboard.navFile
                .get_nearest_area_by_position(vec3Conv(curPos))
                .get_id();
        /*
        AreaId curAreaId = blackboard.nearestNavCell.getNearestArea(curPos);
        if (curAreaId != oldCurAreaId) {
            double curDistance = blackboard.navFile.get_point_to_area_distance_within(vec3Conv(curPos),
                                                                               blackboard.navFile.get_area_by_id(curAreaId));
            double oldDistance = blackboard.navFile.get_point_to_area_distance_within(vec3Conv(curPos),
                                                                               blackboard.navFile.get_area_by_id(oldCurAreaId));
            if (curDistance < oldDistance) {
                blackboard.navFile.get_nearest_area_by_position(vec3Conv(curPos));
            }
            if (oldDistance == 0.) {
                AreaId tmpOldCurAreaId = blackboard.navFile.get_nearest_area_by_position(vec3Conv(curPos)).get_id();
                AreaId tmpCurAreaId = blackboard.nearestNavCell.getNearestArea(curPos);
                blackboard.navFile.get_nearest_area_by_position(vec3Conv(curPos));
            }
            std::cout << "nearest area id cache wrong for pos "
                << state.getClient(treeThinker.csgoId).getFootPosForPlayer().toCSV() <<
                " old area id " << oldCurAreaId << " distance " << oldDistance << " aabb " <<
                blackboard.nearestNavCell.visPoints.getAreaVisPoint(oldCurAreaId).areaCoordinates.min.toCSV() <<
                blackboard.nearestNavCell.visPoints.getAreaVisPoint(oldCurAreaId).areaCoordinates.max.toCSV() <<
                " new area id " << curAreaId << " distance " << curDistance << " aabb " <<
                blackboard.nearestNavCell.visPoints.getAreaVisPoint(curAreaId).areaCoordinates.min.toCSV() <<
                blackboard.nearestNavCell.visPoints.getAreaVisPoint(curAreaId).areaCoordinates.max.toCSV() <<
                " new area connections size " << blackboard.navFile.get_area_by_id(curAreaId).get_connections().size() <<
                std::endl;
        }
         */
        for (const auto & [enemyId, enemyPos] : blackboard.getCommunicatedPlayers(state, treeThinker).positions) {
            AreaId enemyAreaId = blackboard.navFile
                    .get_nearest_area_by_position(vec3Conv(enemyPos.lastSeenFootPos)).get_id();
            int64_t enemyAreaIndex = blackboard.navFile.m_area_ids_to_indices[enemyAreaId];
            AreaBits dangerAreaBits = blackboard.visPoints.getDangerRelativeToSrc(curAreaId);
            double minTimeToVis = csknow::feature_store::maxTimeToVis;
            for (size_t dangerAreaIndex = 0; dangerAreaIndex < dangerAreaBits.size(); dangerAreaIndex++) {
                if (dangerAreaBits[dangerAreaIndex]) {
                    double secondsAway =
                        secondsAwayAtMaxSpeed(blackboard.reachability.getDistance(enemyAreaIndex, dangerAreaIndex));
                    if (secondsAway < COMMUNICATED_ENEMY_RELEVANT_TIME &&
                        relevantCommunicatedEnemies.find(enemyAreaId) == relevantCommunicatedEnemies.end()) {
                        relevantCommunicatedEnemies[enemyId] = enemyPos;
                    }
                    minTimeToVis = std::min(minTimeToVis, secondsAway);
                }
            }

            if (possibleEnemies.find(enemyId) == possibleEnemies.end() && minTimeToVis < COMMUNICATED_ENEMY_RELEVANT_TIME) {
                possibleEnemies[enemyId] = {
                    enemyId, csknow::feature_store::EngagementEnemyState::Communicated, minTimeToVis,
                    INVALID_ID, INVALID_ID
                };
            }
        }

        for (auto & [victimId, possibleEnemy] : possibleEnemies) {
            const ServerState::Client & victimClient = state.getClient(victimId);
            setPossibleEnemyDistances(curClient.getEyePosForPlayer(), curClient.getCurrentViewAngles(),
                                      victimClient.getEyePosForPlayer(), victimClient.getCurrentViewAngles(),
                                      victimClient.duckAmount, possibleEnemy);
            blackboard.featureStorePreCommitBuffer.addEngagementPossibleEnemy(possibleEnemy);
        }

        for (auto & teammateId : state.getPlayersOnTeam(curClient.team)) {
            const auto & teammateClient = state.getClient(teammateId);
            if (teammateId == curClient.csgoId || !teammateClient.isAlive) {
                continue;
            }
            csknow::feature_store::EngagementTeammate engagementTeammate{};
            engagementTeammate.playerId = teammateId;
            setTeammateDistances(curClient.getEyePosForPlayer(), curClient.getCurrentViewAngles(),
                                      teammateClient.getEyePosForPlayer(), teammateClient.getCurrentViewAngles(),
                                      teammateClient.duckAmount, engagementTeammate);
            blackboard.featureStorePreCommitBuffer.addEngagementTeammate(engagementTeammate);
        }

        bool communicatedEnemy = !relevantCommunicatedEnemies.empty();
        return enemyVisible || rememberEnemy || communicatedEnemy;
    }
};

#endif //CSKNOW_ENGAGE_NODE_H
