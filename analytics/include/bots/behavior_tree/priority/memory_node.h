//
// Created by steam on 7/4/22.
//

#ifndef CSKNOW_MEMORY_NODE_H
#define CSKNOW_MEMORY_NODE_H

#include "bots/behavior_tree/node.h"

namespace memory {
    class PerPlayerMemory : public Node {
    public:
        PerPlayerMemory(Blackboard & blackboard) : Node(blackboard, "PerPlayerMemory") { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

}

#endif //CSKNOW_MEMORY_NODE_H
