//
// Created by durst on 5/6/23.
//

#ifndef CSKNOW_COMPUTE_MODEL_NAV_AREA_NODE_H
#define CSKNOW_COMPUTE_MODEL_NAV_AREA_NODE_H

#include "bots/behavior_tree/node.h"
#include "bots/behavior_tree/pathing_node.h"
#include <map>
#include "bots/behavior_tree/condition_helper_node.h"
#include "bots/behavior_tree/priority/model_nav_data.h"

namespace csknow {
    namespace compute_nav_area {
        class ComputeModelNavAreaNode : public Node {
            bool inEngagePath;
        public:
            ComputeModelNavAreaNode(Blackboard & blackboard, bool inEngagePath) :
                Node(blackboard, "ComputeModelNavAreaNode"), inEngagePath(inEngagePath) { };
            void computeDeltaPosProbabilistic(const ServerState & state, Priority & curPriority, CSGOId csgoId,
                                              ModelNavData & modelNavData);
            virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
        };


    }
}

#endif //CSKNOW_COMPUTE_MODEL_NAV_AREA_NODE_H
