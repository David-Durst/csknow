//
// Created by durst on 2/21/23.
//

#ifndef CSKNOW_BEHAVIOR_TREE_LATENT_EVENTS_H
#define CSKNOW_BEHAVIOR_TREE_LATENT_EVENTS_H

#include "queries/moments/engagement.h"
#include "bots/behavior_tree/blackboard.h"
#include "bots/behavior_tree/node.h"
#include "bots/behavior_tree/global/communicate_node.h"
#include "bots/behavior_tree/global/strategy_node.h"
#include "bots/behavior_tree/priority/engage_node.h"

namespace csknow::behavior_tree_latent_events {
    class GlobalQueryNode : public SequenceNode {
    public:
        GlobalQueryNode(Blackboard & blackboard) :
                SequenceNode(blackboard, Node::makeList(
                        make_unique<strategy::CreateOrdersNode>(blackboard),
                        make_unique<communicate::CommunicateTeamMemory>(blackboard)
                ), "GlobalQueryNodes") { };
    };

    class PlayerQueryNode : public SequenceNode {
    public:
        PlayerQueryNode(Blackboard & blackboard) :
                SequenceNode(blackboard, Node::makeList(
                        make_unique<memory::PerPlayerMemory>(blackboard),
                        make_unique<EnemyEngageCheckNode>(blackboard)
                ), "GlobalQueryNodes") { };
    };

    enum class LatentEventType {
        DefusalOffensive,
        DefusalDefense,
        Engagement
    };

    class BehaviorTreeLatentEvents : public QueryResult {
    public:
        vector<RangeIndexEntry> rowIndicesPerRound;
        vector<int64_t> startTickId;
        vector<int64_t> endTickId;
        vector<int64_t> tickLength;
        vector<LatentEventType> eventType;
        vector<int64_t> playerId;
        IntervalIndex eventsPerTick;

        BehaviorTreeLatentEvents() {
            variableLength = false;
            nonTemporal = true;
            overlay = true;
        }

        vector<int64_t> filterByForeignKey(int64_t) override {
            return {};
        }

        void oneLineToCSV(int64_t, std::ostream &) override { }

        [[nodiscard]]
        vector<string> getForeignKeyNames() override {
            return {};
        }

        [[nodiscard]]
        vector<string> getOtherColumnNames() override {
            return {};
        }

        void runQuery(const string & navPath, VisPoints visPoints,
                      const MapMeshResult & mapMeshResult, const ReachableResult & reachability,
                      const DistanceToPlacesResult & distanceToPlaces,
                      const nearest_nav_cell::NearestNavCell & nearestNavCell,
                      const Players & players, const Rounds & rounds, const Ticks & ticks,
                      const PlayerAtTick & playerAtTick,
                      const WeaponFire & weaponFire, const Hurt & hurt);
    };

}

#endif //CSKNOW_BEHAVIOR_TREE_LATENT_EVENTS_H
