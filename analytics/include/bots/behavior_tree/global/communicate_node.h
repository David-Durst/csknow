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
#include "bots/behavior_tree/condition_helper_node.h"
#define WATCHED_DISTANCE 750.
#define RECENTLY_CHECKED_SECONDS 5.
#define DANGER_ATTENTION_SECONDS 1.

namespace communicate {
    namespace spacing {
        class AssignEntryIndexNode : public Node {
        public:
            AssignEntryIndexNode(Blackboard & blackboard) : Node(blackboard, "AssignEntryIndexNode") { };
            virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
        };

        class AssignHoldIndexNode : public Node {
        public:
            AssignHoldIndexNode(Blackboard & blackboard) : Node(blackboard, "AssignHoldIndexNode") { };
            virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
        };
    }

    class AssignSpacingNode : public SequenceNode {
    public:
        AssignSpacingNode(Blackboard & blackboard) :
                SequenceNode(blackboard, Node::makeList(
                        make_unique<spacing::AssignEntryIndexNode>(blackboard),
                        make_unique<spacing::AssignHoldIndexNode>(blackboard)
                ), "CommunicationNode") { };
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
                    make_unique<communicate::AssignSpacingNode>(blackboard),
                    make_unique<communicate::CommunicateTeamMemory>(blackboard),
                    make_unique<communicate::DiffusePositionsNode>(blackboard),
                    make_unique<communicate::PrioritizeDangerAreasNode>(blackboard)
    ), "CommunicationNode") { };
};

#endif //CSKNOW_COMMUNICATE_NODE_H
