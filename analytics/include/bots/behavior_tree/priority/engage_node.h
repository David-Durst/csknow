//
// Created by durst on 6/9/22.
//

#ifndef CSKNOW_ENGAGE_NODE_H
#define CSKNOW_ENGAGE_NODE_H
#include "bots/behavior_tree/node.h"
#include "bots/behavior_tree/pathing_node.h"
#include <map>

namespace engage {
    class SelectTargetNode : public Node {
    public:
        SelectTargetNode(Blackboard & blackboard) : Node(blackboard, "SelectTargetNode") { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    class SelectFireModeNode : public Node {
    public:
        SelectFireModeNode(Blackboard & blackboard) : Node(blackboard, "FireSelectionTaskNode") { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };
}

class EngageNode : public SequenceNode {
public:
    EngageNode(Blackboard & blackboard) :
            SequenceNode(blackboard, Node::makeList(
                                 make_unique<engage::SelectTargetNode>(blackboard),
                                 make_unique<engage::SelectFireModeNode>(blackboard),
                                 make_unique<movement::PathingNode>(blackboard),
                                 make_unique<movement::WaitNode>(blackboard, 0.5)),
                         "EngageNode") { };
};

class EnemyEngageCheckNode : public ConditionDecorator {
public:
    EnemyEngageCheckNode(Blackboard & blackboard) : ConditionDecorator(blackboard,
                                                                        make_unique<EngageNode>(blackboard),
                                                                        "EnemyEngageCheckNode") { };

    virtual bool valid(const ServerState & state, TreeThinker & treeThinker) override {
        bool enemyVisible = state.getVisibleEnemies(treeThinker.csgoId).size() > 0;
        bool rememberEnemy = blackboard.playerToMemory[treeThinker.csgoId].positions.size() > 0;
        bool communicatedEnemy = blackboard.getCommunicatedPlayers(state, treeThinker).positions.size() > 0;
        return enemyVisible || rememberEnemy || communicatedEnemy;
    }
};

#endif //CSKNOW_ENGAGE_NODE_H
