//
// Created by steam on 7/5/22.
//

#ifndef CSKNOW_COMMUNICATE_NODE_H
#define CSKNOW_COMMUNICATE_NODE_H

#include "bots/behavior_tree/priority/memory_node.h"
#include "bots/behavior_tree/node.h"
#include <map>
#include <memory>

namespace communicate {
    class AssignAggressionNode : public Node {
    public:
        AssignAggressionNode(Blackboard & blackboard) : Node(blackboard, "AssignAggressionNode") { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    class CommunicateTeamMemory : public Node {
    public:
        CommunicateTeamMemory(Blackboard & blackboard) : Node(blackboard, "CommunicateTeamMemory") { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };
}

class CommunicateNode : public SequenceNode {
public:
    CommunicateNode(Blackboard & blackboard) :
            SequenceNode(blackboard, Node::makeList(
                    make_unique<communicate::AssignAggressionNode>(blackboard),
                    make_unique<communicate::CommunicateTeamMemory>(blackboard)
            ), "CommunicationNode") { };
};

#endif //CSKNOW_COMMUNICATE_NODE_H
