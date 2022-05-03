//
// Created by durst on 4/26/22.
//

#ifndef CSKNOW_NODE_H
#define CSKNOW_NODE_H

#include "load_save_bot_data.h"
#include "navmesh/nav_file.h"
#include "bots/behavior_tree/order_data.h"
#include "queries/nav_mesh.h"
#include "queries/reachable.h"
using std::map;

struct Blackboard {
    nav_mesh::nav_file navFile;

    // general map data
    ReachableResult reachability;
    map<string, vector<int32_t>> navPlaceToArea;

    // order data
    vector<Order> orders;
    map<CSGOId, int64_t> playerToOrder;


};

struct TreeThinker {
    // constant values across game
    CSGOId csgoId;
};

enum class NodeState {
    Uninitialized,
    Success,
    Failure,
    Running,
    NUM_NODE_STATES
};

class Node {
protected:
    Blackboard & blackboard;
    NodeState nodeState;

public:

    Node(Blackboard & blackboard) : blackboard(blackboard), nodeState(NodeState::Uninitialized) { }
    virtual NodeState exec(const ServerState & state, const TreeThinker & treeThinker);
    virtual void reset() {
        blackboard.orders.clear();
        blackboard.playerToOrder.clear();
        nodeState = NodeState::Uninitialized;
    }
};

class RootNode : public Node {
    Node child;

public:
    RootNode(Blackboard & blackboard, Node node) : Node(blackboard), child(node) { };

    NodeState exec(const ServerState & state, const TreeThinker & treeThinker) override {
        return child.exec(state, treeThinker);
    }
};

class ParSelectorNode : public Node {
    vector<Node> children;

public:
    ParSelectorNode(Blackboard & blackboard, vector<Node> nodes) : Node(blackboard), children(nodes) { };

    NodeState exec(const ServerState & state, const TreeThinker & treeThinker) override {
        for (auto & child : children) {
            if (child.exec(state, treeThinker) == NodeState::Failure) {
                return NodeState::Failure;
            }
        }
        return NodeState::Success;
    }
};

class FirstSuccessSeqSelectorNode : public Node {
    vector<Node> children;

public:
    FirstSuccessSeqSelectorNode(Blackboard & blackboard, vector<Node> nodes) : Node(blackboard), children(nodes) { };

    NodeState exec(const ServerState & state, const TreeThinker & treeThinker) override {
        for (auto & child : children) {
            if (child.exec(state, treeThinker) == NodeState::Success) {
                return NodeState::Success;
            }
        }
        return NodeState::Failure;
    }
};


#endif //CSKNOW_NODE_H
