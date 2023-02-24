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
    };

    class SelectFireModeNode : public Node {
    public:
        SelectFireModeNode(Blackboard & blackboard) : Node(blackboard, "FireSelectionTaskNode") { };
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
                                 make_unique<movement::PathingNode>(blackboard)),
                         "EngageNode") { };
};

class EnemyEngageCheckNode : public ConditionDecorator {
public:
    EnemyEngageCheckNode(Blackboard & blackboard) : ConditionDecorator(blackboard,
                                                                        make_unique<EngageNode>(blackboard),
                                                                        "EnemyEngageCheckNode") { };

    virtual bool valid(const ServerState & state, TreeThinker & treeThinker) override {
        bool enemyVisible = !state.getVisibleEnemies(treeThinker.csgoId).empty();
        bool rememberEnemy = !blackboard.playerToMemory[treeThinker.csgoId].positions.empty();

        // select only relevant communicated enemies - those that are near a danger area
        map<CSGOId, EnemyPositionMemory> & relevantCommunicatedEnemies = blackboard.playerToRelevantCommunicatedEnemies[treeThinker.csgoId];
        relevantCommunicatedEnemies.clear();
        Vec3 curPos = state.getClient(treeThinker.csgoId).getFootPosForPlayer();
        AreaId curAreaId = blackboard.nearestNavCell.getNearestArea(curPos);
        AreaId oldCurAreaId = blackboard.navFile
                .get_nearest_area_by_position(vec3Conv(curPos))
                .get_id();
        if (curAreaId != oldCurAreaId) {
            double curDistance = blackboard.navFile.get_point_to_area_distance_within(vec3Conv(curPos),
                                                                               blackboard.navFile.get_area_by_id(curAreaId));
            double oldDistance = blackboard.navFile.get_point_to_area_distance_within(vec3Conv(curPos),
                                                                               blackboard.navFile.get_area_by_id(oldCurAreaId));
            Vec3 testPos{-560.679993,1998.670044,-117.599998};
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
        for (const auto & [enemyId, enemyPos] : blackboard.getCommunicatedPlayers(state, treeThinker).positions) {
            AreaId enemyAreaId = blackboard.navFile
                    .get_nearest_area_by_position(vec3Conv(enemyPos.lastSeenFootPos)).get_id();
            int64_t enemyAreaIndex = blackboard.navFile.m_area_ids_to_indices[enemyAreaId];
            AreaBits dangerAreaBits = blackboard.visPoints.getDangerRelativeToSrc(curAreaId);
            for (size_t dangerAreaIndex = 0; dangerAreaIndex < dangerAreaBits.size(); dangerAreaIndex++) {
                if (dangerAreaBits[dangerAreaIndex] &&
                    secondsAwayAtMaxSpeed(blackboard.reachability.getDistance(enemyAreaIndex, dangerAreaIndex)) < COMMUNICATED_ENEMY_RELEVANT_TIME) {
                    relevantCommunicatedEnemies[enemyId] = enemyPos;
                    break;
                }
            }
        }
        bool communicatedEnemy = !relevantCommunicatedEnemies.empty();
        return enemyVisible || rememberEnemy || communicatedEnemy;
    }
};

#endif //CSKNOW_ENGAGE_NODE_H
