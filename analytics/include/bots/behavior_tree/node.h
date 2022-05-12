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

    // general map data
    ReachableResult reachability;
    map<uint32_t, map<uint32_t, double>> distanceMatrix;
    map<string, vector<uint32_t>> navPlaceToArea;

    // all player data
    map<CSGOId, TreeThinker> treeThinkers;

    // order data
    int32_t planRoundNumber = -1;
    vector<Order> orders;
    map<CSGOId, int64_t> playerToOrder;

    // priority data
    map<CSGOId, Priority> playerToPriority;

    // implementation data
    map<CSGOId, Path> playerToPath;

    // action data
    map<CSGOId, Action> playerToAction;

    string getPlayerPlace(Vec3 pos) {
        return navFile.get_place(navFile.get_nearest_area_by_position(vec3Conv(pos)).m_place);
    }

    void computeDistanceMatrix();
    double getDistance(uint32_t srcArea, uint32_t dstArea) {
        return computeDistance(vec3tConv(navFile.get_area_by_id_fast(srcArea).get_center()),
                               vec3tConv(navFile.get_area_by_id_fast(dstArea).get_center()));
    }

    Blackboard(string navPath) : navFile(navPath.c_str()) {
                                 //reachability(queryReachable(queryMapMesh(navFile))) {
        for (const auto & area : navFile.m_areas) {
            navPlaceToArea[navFile.get_place(area.m_place)].push_back(area.get_id());
        }
    }

};

enum class NodeState {
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
    virtual void reset() {
        blackboard.orders.clear();
        blackboard.playerToOrder.clear();
        playerNodeState.clear();
    }

    virtual PrintState printState(const ServerState & state, CSGOId playerId) const {
        string stateString;
        if (playerNodeState.find(playerId) == playerNodeState.end()) {
            stateString = "Uninitialized";
        }
        else {
            switch (playerNodeState.find(playerId)->second) {
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
};

class ParSelectorNode : public Node {
    vector<Node::Ptr> children;

public:
    ParSelectorNode(Blackboard & blackboard, vector<Node::Ptr> && nodes, string name) :
        Node(blackboard, name), children(std::move(nodes)) { };

    NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        bool anyChildrenRunning = false, anyChildrenSuccess;
        for (auto & child : children) {
            NodeState childNodeState = child->exec(state, treeThinker);
            if (childNodeState == NodeState::Running) {
                anyChildrenRunning = true;
            }
            else if (childNodeState == NodeState::Success) {
                anyChildrenSuccess = true;
            }
        }
        if (anyChildrenRunning) {
            playerNodeState[treeThinker.csgoId] = NodeState::Running;
        }
        else if (anyChildrenSuccess) {
            playerNodeState[treeThinker.csgoId] = NodeState::Success;
        }
        else {
            playerNodeState[treeThinker.csgoId] = NodeState::Failure;
        }
        return playerNodeState[treeThinker.csgoId];
    }

    void reset() override {
        for (auto & child : children) {
            child.reset();
        }
        Node::reset();
    }

    virtual PrintState printState(const ServerState & state, CSGOId playerId) const override {
        PrintState printState = Node::printState(state, playerId);
        for (const auto & child : children) {
            printState.childrenStates.push_back(child->printState(state, playerId));
        }
        return printState;
    }
};

class FirstNonFailSeqSelectorNode : public Node {
protected:
    vector<Node::Ptr> children;

public:
    FirstNonFailSeqSelectorNode(Blackboard & blackboard, vector<Node::Ptr> nodes, string name) :
        Node(blackboard, name), children(std::move(nodes)) { };

    NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        for (auto & child : children) {
            NodeState childNodeState = child->exec(state, treeThinker);
            if (childNodeState != NodeState::Failure) {
                playerNodeState[treeThinker.csgoId] = childNodeState;
                return playerNodeState[treeThinker.csgoId];
            }
        }
        playerNodeState[treeThinker.csgoId] = NodeState::Failure;
        return NodeState::Failure;
    }

    void reset() override {
        for (auto & child : children) {
            child.reset();
        }
        Node::reset();
    }

    virtual PrintState printState(const ServerState & state, CSGOId playerId) const override {
        PrintState printState = Node::printState(state, playerId);
        for (const auto & child : children) {
            printState.childrenStates.push_back(child->printState(state, playerId));
        }
        return printState;
    }
};


#endif //CSKNOW_NODE_H
