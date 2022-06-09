//
// Created by durst on 4/26/22.
//

#ifndef CSKNOW_NODE_H
#define CSKNOW_NODE_H

#include "load_save_bot_data.h"
#include "geometryNavConversions.h"
#include "navmesh/nav_file.h"
#include "bots/behavior_tree/order_data.h"
#include "bots/behavior_tree/priority/priority_data.h"
#include "bots/behavior_tree/implementation_data.h"
#include "bots/behavior_tree/action_data.h"
#include "queries/nav_mesh.h"
#include "queries/reachable.h"
#include <memory>
#include <random>
using std::map;
using std::make_unique;

enum class AggressiveType {
    Push,
    Bait,
    NUM_AGGESSIVE_TYPE
};

struct EngagementParams {
    double standDistance;
    double moveDistance;
    double burstDistance;
    double sprayDistance;
};

struct TreeThinker {
    // constant values across game
    CSGOId csgoId;
    AggressiveType aggressiveType;
    EngagementParams engagementParams;

    int64_t orderWaypointIndex;
    int64_t orderGrenadeIndex;
};

struct Blackboard {
    nav_mesh::nav_file navFile;
    ServerState lastFrameState;

    // helpers
    std::random_device rd;
    std::mt19937 gen;

    // general map data
    ReachableResult reachability;
    map<uint32_t, map<uint32_t, double>> distanceMatrix;
    map<string, vector<uint32_t>> navPlaceToArea;

    // all player data
    map<CSGOId, TreeThinker> playerToTreeThinkers;

    // order data
    int32_t planRoundNumber = -1;
    vector<Order> orders;
    map<CSGOId, int64_t> playerToOrder;

    // priority data
    map<CSGOId, Priority> playerToPriority;

    // implementation data
    map<CSGOId, Path> playerToPath;
    map<CSGOId, uint32_t> playerToCurNavAreaId;

    // action data
    map<CSGOId, Action> playerToAction;
    map<CSGOId, Action> lastPlayerToAction;
    map<CSGOId, PIDState> playerToPIDStateX, playerToPIDStateY;
    std::uniform_real_distribution<> aimDis;

    string getPlayerPlace(Vec3 pos) {
        return navFile.get_place(navFile.get_nearest_area_by_position(vec3Conv(pos)).m_place);
    }

    void computeDistanceMatrix();
    double getDistance(uint32_t srcArea, uint32_t dstArea) {
        return computeDistance(vec3tConv(navFile.get_area_by_id_fast(srcArea).get_center()),
                               vec3tConv(navFile.get_area_by_id_fast(dstArea).get_center()));
    }

    Blackboard(string navPath) : navFile(navPath.c_str()), gen(rd()), aimDis(0., 2.0) {
                                 //reachability(queryReachable(queryMapMesh(navFile))) {
        for (const auto & area : navFile.m_areas) {
            navPlaceToArea[navFile.get_place(area.m_place)].push_back(area.get_id());
        }
    }

};

enum class NodeState {
    Uninitialized,
    Success,
    Failure,
    Running,
    NUM_NODE_STATES
};

struct PrintState {
    vector<PrintState> childrenStates;
    vector<string> curState;

    void getStateInner(size_t depth, stringstream & ss) const {
        for (const auto & curStateLine : curState) {
            for (size_t i = 0; i < depth; i++) {
                ss << "  ";
            }
            ss << curStateLine;
            ss << std::endl;
        }
        for (const auto & childState : childrenStates) {
            childState.getStateInner(depth + 1, ss);
        }
    }

    string getState() const {
        stringstream ss;
        getStateInner(0, ss);
        return ss.str();
    }
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

class SequenceNode : public Node {
    vector<Node::Ptr> children;
    map<CSGOId, size_t> curChildIndex;

public:
    SequenceNode(Blackboard & blackboard, vector<Node::Ptr> && nodes, string name) :
        Node(blackboard, name), children(std::move(nodes)) { };

    NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        if (playerNodeState.find(treeThinker.csgoId) == playerNodeState.end() ||
            playerNodeState[treeThinker.csgoId] != NodeState::Running) {
            curChildIndex[treeThinker.csgoId] = 0;
            restart(treeThinker);
        }
        playerNodeState[treeThinker.csgoId] = NodeState::Running;
        while (true) {
            NodeState childNodeState = children[curChildIndex[treeThinker.csgoId]]->exec(state, treeThinker);
            if (childNodeState == NodeState::Success) {
                curChildIndex[treeThinker.csgoId]++;
                if (curChildIndex[treeThinker.csgoId] == children.size() - 1) {
                    break;
                }
            }
            else if (childNodeState == NodeState::Running) {
                playerNodeState[treeThinker.csgoId] = NodeState::Running;
                return playerNodeState[treeThinker.csgoId];
            }
            else if (childNodeState == NodeState::Failure) {
                playerNodeState[treeThinker.csgoId] = NodeState::Failure;
                return playerNodeState[treeThinker.csgoId];
            }
        }
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }

    void clearState() override {
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

class SelectorNode : public Node {
protected:
    vector<Node::Ptr> children;
    map<CSGOId, size_t> curChildIndex;

public:
    SelectorNode(Blackboard & blackboard, vector<Node::Ptr> nodes, string name) :
        Node(blackboard, name), children(std::move(nodes)) { };

    NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        if (playerNodeState.find(treeThinker.csgoId) == playerNodeState.end() ||
            playerNodeState[treeThinker.csgoId] != NodeState::Running) {
            curChildIndex[treeThinker.csgoId] = 0;
            restart(treeThinker);
        }
        playerNodeState[treeThinker.csgoId] = NodeState::Running;
        while (true) {
            NodeState childNodeState = children[curChildIndex[treeThinker.csgoId]]->exec(state, treeThinker);
            if (childNodeState == NodeState::Failure) {
                curChildIndex[treeThinker.csgoId]++;
                if (curChildIndex[treeThinker.csgoId] == children.size() - 1) {
                    break;
                }
            }
            else if (childNodeState == NodeState::Running) {
                playerNodeState[treeThinker.csgoId] = NodeState::Running;
                return playerNodeState[treeThinker.csgoId];
            }
            else if (childNodeState == NodeState::Success) {
                playerNodeState[treeThinker.csgoId] = NodeState::Success;
                return playerNodeState[treeThinker.csgoId];
            }
        }
        playerNodeState[treeThinker.csgoId] = NodeState::Failure;
        return playerNodeState[treeThinker.csgoId];
    }

    void clearState() override {
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

class ConditionDecorator : public Node {
protected:
    Node::Ptr child;

public:
    ConditionDecorator(Blackboard & blackboard, Node::Ptr nodes, string name) :
        Node(blackboard, name), child (std::move(nodes)) { };
}

#endif //CSKNOW_NODE_H
