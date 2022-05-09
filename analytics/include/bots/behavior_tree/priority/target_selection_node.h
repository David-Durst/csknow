//
// Created by durst on 5/5/22.
//

#ifndef CSKNOW_TARGET_SELECTION_NODE_H
#define CSKNOW_TARGET_SELECTION_NODE_H

#include "bots/behavior_tree/node.h"

class TargetSelectionTaskNode : public Node {
public:
    TargetSelectionTaskNode(Blackboard & blackboard) : Node(blackboard) { };
    NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
};

#endif //CSKNOW_TARGET_SELECTION_NODE_H
