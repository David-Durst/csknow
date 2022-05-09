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
using std::map;

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
    map<string, vector<uint32_t>> navPlaceToArea;

    // all player data
    map<CSGOId, TreeThinker> treeThinkers;

    // order data
    vector<Order> orders;
    map<CSGOId, int64_t> playerToOrder;

    // priority data
    map<CSGOId, Priority> playerToPriority;

    // implementation data
    map<CSGOId, Path> playerToPath;

    // action data
    map<CSGOId, Action> playerToAction;

    string getPlayerPlace(Vec3 pos) {
        return navFile.m_places[navFile.get_nearest_area_by_position(vec3Conv(pos)).m_place];
    }

    Blackboard(string navPath) : navFile(navPath.c_str()),
        reachability(queryReachable(queryMapMesh(navFile))) {
        for (const auto & area : navFile.m_areas) {
            navPlaceToArea[navFile.m_places[area.m_place]].push_back(area.get_id());
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

class Node {
public:
    Blackboard & blackboard;
    NodeState nodeState;


    Node(Blackboard & blackboard) : blackboard(blackboard), nodeState(NodeState::Uninitialized) { }
    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker);
    virtual void reset() {
        blackboard.orders.clear();
        blackboard.playerToOrder.clear();
        nodeState = NodeState::Uninitialized;
    }

    uint32_t getNearestAreaInNextPlace(const ServerState & state, const TreeThinker & treeThinker, string nextPlace);
};

    class RootNode : public Node {
    Node child;

public:
    RootNode(Blackboard & blackboard, Node node) : Node(blackboard), child(node) { };

    NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        return child.exec(state, treeThinker);
    }
};

class ParSelectorNode : public Node {
    vector<Node> children;

public:
    ParSelectorNode(Blackboard & blackboard, vector<Node> nodes) : Node(blackboard), children(nodes) { };

    NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        bool anyChildrenRunning = false, anyChildrenSuccess;
        for (auto & child : children) {
            NodeState childNodeState = child.exec(state, treeThinker);
            if (childNodeState == NodeState::Running) {
                anyChildrenRunning = true;
            }
            else if (childNodeState == NodeState::Success) {
                anyChildrenSuccess = true;
            }
        }
        if (anyChildrenRunning) {
            nodeState = NodeState::Running;
        }
        else if (anyChildrenSuccess) {
            nodeState = NodeState::Success;
        }
        else {
            nodeState = NodeState::Failure;
        }
        return nodeState;
    }

    void reset() override {
        for (auto & child : children) {
            child.reset();
        }
        Node::reset();
    }
};

class FirstNonFailSeqSelectorNode : public Node {
protected:
    vector<Node> children;

public:
    FirstNonFailSeqSelectorNode(Blackboard & blackboard, vector<Node> nodes) : Node(blackboard), children(nodes) { };

    NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        for (auto & child : children) {
            NodeState childNodeState = child.exec(state, treeThinker);
            if (childNodeState != NodeState::Failure) {
                nodeState = childNodeState;
                return nodeState;
            }
        }
        nodeState = NodeState::Failure;
        return NodeState::Failure;
    }

    void reset() override {
        for (auto & child : children) {
            child.reset();
        }
        Node::reset();
    }
};


#endif //CSKNOW_NODE_H
