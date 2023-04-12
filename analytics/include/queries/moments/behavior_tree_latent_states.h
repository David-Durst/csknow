//
// Created by durst on 2/21/23.
//

#ifndef CSKNOW_BEHAVIOR_TREE_LATENT_STATES_H
#define CSKNOW_BEHAVIOR_TREE_LATENT_STATES_H

#include <variant>
#include "queries/moments/engagement.h"
#include "bots/behavior_tree/blackboard.h"
#include "bots/behavior_tree/node.h"
#include "bots/behavior_tree/global/communicate_node.h"
#include "bots/behavior_tree/global/strategy_node.h"
#include "bots/behavior_tree/priority/engage_node.h"

namespace csknow::behavior_tree_latent_states {
    class GlobalQueryNode : public SequenceNode {
    public:
        GlobalQueryNode(Blackboard & blackboard) :
                SequenceNode(blackboard, Node::makeList(
                        make_unique<strategy::CreateOrdersNode>(blackboard),
                        make_unique<strategy::AssignPlayersToOrders>(blackboard),
                        make_unique<communicate::CommunicateTeamMemory>(blackboard)
                ), "GlobalQueryNodes") { };
    };

    class ClearPriorityNode : public Node {
    public:
        ClearPriorityNode(Blackboard & blackboard) : Node(blackboard, "ClearPriorityNode") { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    class PriorityDecisionNode : public SelectorNode {
    public:
        PriorityDecisionNode(Blackboard & blackboard) :
            SelectorNode(blackboard, Node::makeList(
                             make_unique<EnemyEngageCheckNode>(blackboard),
                             make_unique<ClearPriorityNode>(blackboard)),
                         "PriorityDecisionNode") { };
    };

    class PlayerQueryNode : public ParallelFirstNode {
    public:
        PlayerQueryNode(Blackboard & blackboard) :
                ParallelFirstNode(blackboard, Node::makeList(
                        make_unique<memory::PerPlayerMemory>(blackboard),
                        make_unique<PriorityDecisionNode>(blackboard)
                ), "GlobalQueryNodes") { };
    };

    enum class LatentStateType {
        Order, // since order changes are symmetric (everoyne knows the events), only need 1 state for ct and t
        Engagement
    };

    struct OrderStatePayload { };
    struct EngagementStatePayload {
        CSGOId sourceId, targetId;
    };

    typedef std::variant<OrderStatePayload, EngagementStatePayload> StatePayload;

    class BehaviorTreeLatentStates : public QueryResult {
    public:
        vector<RangeIndexEntry> rowIndicesPerRound;
        vector<int64_t> startTickId;
        vector<int64_t> endTickId;
        vector<int64_t> tickLength;
        vector<LatentStateType> latentStateType;
        vector<StatePayload> statePayload;
        IntervalIndex eventsPerTick;
        feature_store::FeatureStoreResult featureStoreResult;

        BehaviorTreeLatentStates(const Ticks & ticks, const PlayerAtTick & playerAtTick,
                                 const std::vector<csknow::orders::QueryOrder> & orders) :
            featureStoreResult(ticks.size, playerAtTick.size, orders) {
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
                      const csknow::orders::OrdersResult & ordersResult,
                      const Players & players, const Games & games, const Rounds & rounds,
                      const Ticks & ticks, const PlayerAtTick & playerAtTick,
                      const WeaponFire & weaponFire, const Hurt & hurt,
                      const Plants & plants, const Defusals & defusals,
                      const EngagementResult & acausalEngagementResult);
    };

}

#endif //CSKNOW_BEHAVIOR_TREE_LATENT_STATES_H
