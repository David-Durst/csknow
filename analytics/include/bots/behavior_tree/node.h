//
// Created by durst on 4/26/22.
//

#ifndef CSKNOW_NODE_H
#define CSKNOW_NODE_H

#include "load_save_bot_data.h"
#include "geometryNavConversions.h"
#include "navmesh/nav_file.h"
#include "bots/behavior_tree/global/order_data.h"
#include "bots/behavior_tree/priority/priority_data.h"
#include "bots/behavior_tree/pathing_data.h"
#include "bots/behavior_tree/action_data.h"
#include "bots/behavior_tree/blackboard.h"
#include "queries/nav_mesh.h"
#include "queries/reachable.h"
#include <memory>
#include <random>
using std::map;
using std::make_unique;

enum class NodeState {
    Uninitialized,
    Success,
    Failure,
    Running,
    NUM_NODE_STATES
};

class Node {
public:
    using Ptr = std::unique_ptr<Node>;
    Blackboard & blackboard;
    map<CSGOId, NodeState> playerNodeState;
    string name;


    Node(Blackboard & blackboard, string name) : blackboard(blackboard), playerNodeState({}), name(name) { }
    virtual ~Node() { };
    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) = 0;
    virtual void clearState() {
        blackboard.orders.clear();
        blackboard.playerToOrder.clear();
        playerNodeState.clear();
    }

    virtual void restart(const TreeThinker & treeThinker) {
        playerNodeState[treeThinker.csgoId] = NodeState::Uninitialized;
    }

    virtual PrintState printState(const ServerState & state, CSGOId playerId) const {
        string stateString;
        if (playerNodeState.find(playerId) == playerNodeState.end()) {
            stateString = "Uninitialized";
        }
        else {
            switch (playerNodeState.find(playerId)->second) {
                case NodeState::Uninitialized:
                    stateString = "Uninitialized";
                    break;
                case NodeState::Success:
                    stateString = "Success";
                    break;
                case NodeState::Failure:
                    stateString = "Failure";
                    break;
                case NodeState::Running:
                    stateString = "Running";
                    break;
                default:
                    stateString = "BAD NODE STATE";
            }
        }
        return {{}, {name + ": " + stateString}};
    }

    template <typename ...Args>
    static vector<Node::Ptr> makeList(Args ...args)
    {
        vector<Node::Ptr> nodes;
        constexpr size_t n = sizeof...(Args);
        nodes.reserve(n);

        (
            nodes.emplace_back(std::move(args)), ...
        );
    
        return nodes;
    }

    uint32_t getNearestAreaInNextPlace(const ServerState & state, const TreeThinker & treeThinker, string nextPlace);
    uint32_t getRandomAreaInNextPlace(const ServerState & state, string nextPlace);
};

class CollectionNode : public Node {
protected:
    vector<Node::Ptr> children;
    map<CSGOId, size_t> curChildIndex;

public:
    CollectionNode(Blackboard & blackboard, vector<Node::Ptr> && nodes, string name) :
        Node(blackboard, name), children(std::move(nodes)) { };

    virtual void clearState() override {
        for (auto & child : children) {
            child->clearState();
        }
        Node::clearState();
    }

    virtual void restart(const TreeThinker & treeThinker) override {
        for (auto & child : children) {
            child->restart(treeThinker);
        }
        Node::restart(treeThinker);
    }

    virtual PrintState printState(const ServerState & state, CSGOId playerId) const override {
        PrintState printState = Node::printState(state, playerId);
        for (const auto & child : children) {
            printState.childrenStates.push_back(child->printState(state, playerId));
        }
        return printState;
    }
};

class SequenceNode : public CollectionNode {
public:
    SequenceNode(Blackboard & blackboard, vector<Node::Ptr> && nodes, string name = "Sequence") :
        CollectionNode(blackboard, std::move(nodes), name) { };

    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        if (playerNodeState.find(treeThinker.csgoId) == playerNodeState.end() ||
            playerNodeState[treeThinker.csgoId] != NodeState::Running) {
            curChildIndex[treeThinker.csgoId] = 0;
            restart(treeThinker);
        }
        playerNodeState[treeThinker.csgoId] = NodeState::Running;
        while (curChildIndex[treeThinker.csgoId] < children.size()) {
            NodeState childNodeState = children[curChildIndex[treeThinker.csgoId]]->exec(state, treeThinker);
            if (childNodeState == NodeState::Success) {
                curChildIndex[treeThinker.csgoId]++;
            }
            else if (childNodeState == NodeState::Running) {
                playerNodeState[treeThinker.csgoId] = NodeState::Running;
                return playerNodeState[treeThinker.csgoId];
            }
            else if (childNodeState == NodeState::Failure) {
                playerNodeState[treeThinker.csgoId] = NodeState::Failure;
                return playerNodeState[treeThinker.csgoId];
            }
            else if (childNodeState == NodeState::Uninitialized) {
                throw std::runtime_error("child node returned state uninitialized");
            }
        }
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
};

class SelectorNode : public CollectionNode {
public:
    SelectorNode(Blackboard & blackboard, vector<Node::Ptr> && nodes, string name = "Selector") :
        CollectionNode(blackboard, std::move(nodes), name) { };

    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        if (playerNodeState.find(treeThinker.csgoId) == playerNodeState.end() ||
            playerNodeState[treeThinker.csgoId] != NodeState::Running) {
            curChildIndex[treeThinker.csgoId] = 0;
            restart(treeThinker);
        }
        playerNodeState[treeThinker.csgoId] = NodeState::Running;
        while (curChildIndex[treeThinker.csgoId] < children.size()) {
            NodeState childNodeState = children[curChildIndex[treeThinker.csgoId]]->exec(state, treeThinker);
            if (childNodeState == NodeState::Failure) {
                curChildIndex[treeThinker.csgoId]++;
            }
            else if (childNodeState == NodeState::Running) {
                playerNodeState[treeThinker.csgoId] = NodeState::Running;
                return playerNodeState[treeThinker.csgoId];
            }
            else if (childNodeState == NodeState::Success) {
                playerNodeState[treeThinker.csgoId] = NodeState::Success;
                return playerNodeState[treeThinker.csgoId];
            }
            else if (childNodeState == NodeState::Uninitialized) {
                throw std::runtime_error("child node returned state uninitialized");
            }
        }
        playerNodeState[treeThinker.csgoId] = NodeState::Failure;
        return playerNodeState[treeThinker.csgoId];
    }
};

class ParallelAndNode : public CollectionNode {
public:
    ParallelAndNode(Blackboard & blackboard, vector<Node::Ptr> && nodes, string name = "ParallelAnd") :
            CollectionNode(blackboard, std::move(nodes), name) { };

    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        if (playerNodeState.find(treeThinker.csgoId) == playerNodeState.end() ||
            playerNodeState[treeThinker.csgoId] != NodeState::Running) {
            restart(treeThinker);
        }
        bool stillRunning = false;
        for (size_t i = 0; i < children.size(); i++) {
            // skip all children that already succeeded
            if (children[i]->playerNodeState[treeThinker.csgoId] == NodeState::Success) {
                continue;
            }
            NodeState childNodeState = children[i]->exec(state, treeThinker);
            if (childNodeState == NodeState::Success) {
                continue;
            }
            else if (childNodeState == NodeState::Running) {
                stillRunning = true;
                continue;
            }
            else if (childNodeState == NodeState::Failure) {
                playerNodeState[treeThinker.csgoId] = NodeState::Failure;
                return playerNodeState[treeThinker.csgoId];
            }
            else if (childNodeState == NodeState::Uninitialized) {
                throw std::runtime_error("child node returned state uninitialized");
            }
        }
        playerNodeState[treeThinker.csgoId] = stillRunning ? NodeState::Running : NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
};

class ParallelFirstNode : public CollectionNode {
public:
    ParallelFirstNode(Blackboard & blackboard, vector<Node::Ptr> && nodes, string name = "ParallelFirst") :
            CollectionNode(blackboard, std::move(nodes), name) { };

    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        if (playerNodeState.find(treeThinker.csgoId) == playerNodeState.end() ||
            playerNodeState[treeThinker.csgoId] != NodeState::Running) {
            restart(treeThinker);
        }
        for (size_t i = 0; i < children.size(); i++) {
            NodeState childNodeState = children[i]->exec(state, treeThinker);
            if (childNodeState == NodeState::Success) {
                playerNodeState[treeThinker.csgoId] = NodeState::Success;
                return playerNodeState[treeThinker.csgoId];
            }
            else if (childNodeState == NodeState::Running) {
                continue;
            }
            else if (childNodeState == NodeState::Failure) {
                playerNodeState[treeThinker.csgoId] = NodeState::Failure;
                return playerNodeState[treeThinker.csgoId];
            }
            else if (childNodeState == NodeState::Uninitialized) {
                throw std::runtime_error("child node returned state uninitialized");
            }
        }
        playerNodeState[treeThinker.csgoId] = NodeState::Running;
        return playerNodeState[treeThinker.csgoId];
    }
};

class ConditionDecorator : public Node {
protected:
    Node::Ptr child;

public:
    ConditionDecorator(Blackboard & blackboard, Node::Ptr && node, string name = "Condition") :
        Node(blackboard, name), child(std::move(node)) { };

    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        if (valid(state, treeThinker)) {
            playerNodeState[treeThinker.csgoId] = child->exec(state, treeThinker);
        }
        else {
            child->restart(treeThinker);
            playerNodeState[treeThinker.csgoId] = NodeState::Failure;
        }
        return playerNodeState[treeThinker.csgoId];
    }

    virtual bool valid(const ServerState & state, TreeThinker &treeThinker) = 0;

    virtual PrintState printState(const ServerState & state, CSGOId playerId) const override {
        PrintState printState = Node::printState(state, playerId);
        printState.childrenStates.push_back(child->printState(state, playerId));
        return printState;
    }
};

class RepeatDecorator : public Node {
protected:
    Node::Ptr child;
    bool repeatUntilSuccess;

public:
    RepeatDecorator(Blackboard & blackboard, Node::Ptr && node, bool repeatUntilSuccess, string name = "Repeat") :
            Node(blackboard, name), child(std::move(node)), repeatUntilSuccess(repeatUntilSuccess) { };

    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        if (playerNodeState.find(treeThinker.csgoId) == playerNodeState.end() ||
            playerNodeState[treeThinker.csgoId] != NodeState::Running) {
            restart(treeThinker);
        }
        NodeState childState = child->exec(state, treeThinker);
        if (repeatUntilSuccess) {
            playerNodeState[treeThinker.csgoId] = childState == NodeState::Success ? NodeState::Success : NodeState::Running;
        }
        else {
            playerNodeState[treeThinker.csgoId] = childState == NodeState::Failure ? NodeState::Failure : NodeState::Running;
        }
        return playerNodeState[treeThinker.csgoId];
    }

    virtual PrintState printState(const ServerState & state, CSGOId playerId) const override {
        PrintState printState = Node::printState(state, playerId);
        printState.childrenStates.push_back(child->printState(state, playerId));
        return printState;
    }
};

#endif //CSKNOW_NODE_H
