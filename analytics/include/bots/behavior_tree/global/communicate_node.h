//
// Created by steam on 7/5/22.
//

#ifndef CSKNOW_COMMUNICATE_NODE_H
#define CSKNOW_COMMUNICATE_NODE_H

#include "bots/behavior_tree/priority/memory_node.h"
#include "bots/behavior_tree/global/possible_nav_areas.h"
#include "bots/behavior_tree/node.h"
#include <map>
#include <memory>
#define WATCHED_DISTANCE 750.

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

    class DiffusePositionsNode : public Node {
        int32_t diffuseRoundNumber = -1;
    public:
        DiffusePositionsNode(Blackboard & blackboard) : Node(blackboard, "DiffusePositionsNode") { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    class PrioritizeDangerAreasNode : public Node {
    public:
        PrioritizeDangerAreasNode(Blackboard & blackboard) : Node(blackboard, "PrioritizeDangerAreas") { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };
}

class CommunicateNode : public SequenceNode {
public:
    CommunicateNode(Blackboard & blackboard) :
            SequenceNode(blackboard, Node::makeList(
                    make_unique<communicate::AssignAggressionNode>(blackboard),
                    make_unique<communicate::CommunicateTeamMemory>(blackboard),
                    make_unique<communicate::DiffusePositionsNode>(blackboard)
                    //make_unique<communicate::PrioritizeDangerAreasNode>(blackboard)
    ), "CommunicationNode") { };
};

#endif //CSKNOW_COMMUNICATE_NODE_H
