//
// Created by durst on 5/5/22.
//

#ifndef CSKNOW_ENGAGE_NODE_H
#define CSKNOW_ENGAGE_NODE_H

namespace engage {
    class TargetSelectionTaskNode : public Node {
    public:
        TargetSelectionTaskNode(Blackboard & blackboard) : Node(blackboard) { };
        NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    class NotMoveShootTaskNode : public Node {
    public:
        NotMoveShootTaskNode(Blackboard & blackboard) : Node(blackboard) { };
        NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    class MoveShootTaskNode : public Node {
    public:
        StandShootTaskNode(Blackboard & blackboard) : Node(blackboard) { };
        NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };
}

class EngageSeqSelectorNode : public ParSelectorNode {
    vector<Node> nodes;
    SeqSelectorNode(Blackboard & blackboard) :
            FirstSuccessSeqSelectorNode(blackboard, { engage::TargetSelectionTaskNode(blackboard),
                                                      engage::NotMoveShootTaskNode(blackboard),
                                                      engage::MoveShootTaskNode(blackboard)}) { };

};

#endif //CSKNOW_ENGAGE_NODE_H
