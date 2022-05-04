//
// Created by durst on 5/3/22.
//

#include "bots/behavior_tree/priority/follow_order_node.h"
#include "bots/behavior_tree/priority/priority_helpers.h"

namespace follow {
    NodeState PushTaskNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        const Order & curOrder = blackboard.orders[blackboard.playerToOrder[state.csgoIdToCSKnowId[treeThinker.csgoId]]];
        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];

        if (this->nodeState != NodeState::Running) {
            treeThinker.orderWaypointIndex++;
            // done with waypoints if index is over max, then just return success until getting new order
            if (treeThinker.orderWaypointIndex < curOrder.waypoints.size()) {
                moveToWaypoint(*this, state, treeThinker, curOrder, curPriority);
                this->nodeState = NodeState::Running;
            }
            else {
                this->nodeState = NodeState::Success;
            }
        }
        else {
            string curPlace = blackboard.getPlayerPlace(state.clients[state.csgoIdToCSKnowId[treeThinker.csgoId]].getFootPosForPlayer());
            finishWaypoint(*this, state, treeThinker, curOrder, curPriority, curPlace);
        }

        return this->nodeState;
    }
}
