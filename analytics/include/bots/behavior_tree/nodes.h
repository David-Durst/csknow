//
// Created by durst on 4/26/22.
//

#ifndef CSKNOW_NODES_H
#define CSKNOW_NODES_H

#include "load_save_bot_data.h"
#include "navmesh/nav_file.h"
#include <vector>
using std::vector;

struct Blackboard {
    nav_mesh::nav_file navFile;
};

struct TreeThinker {
    // constant values across game
    CSGOId csgoId;
    nav_mesh::nav_file navFile;
};

class Node {
    int64_t index;
    Blackboard & blackboard;

public:
    Node(Blackboard & blackboard) : blackboard(blackboard) { }

    virtual bool relevant(const ServerState & state, const TreeThinker & treeThinker);
    virtual void exec(const ServerState & state, const TreeThinker & treeThinker);
    virtual void onEntry(const TreeThinker & treeThinker) { }
    virtual void onExit(const TreeThinker & treeThinker) { }
};

class RootNode : Node {
    Node child;

public:
    RootNode(Blackboard & blackboard, Node node) : Node(blackboard), child(node) { };

    bool relevant(const ServerState &state, const TreeThinker & treeThinker) override {
        return true;
    }

    void exec(const ServerState & state, const TreeThinker & treeThinker) override {
        child.exec(state, treeThinker);
    }
};

class ParSelectorNode : Node {
    vector<Node> children;

public:
    ParSelectorNode(Blackboard & blackboard, vector<Node> nodes) : Node(blackboard), children(nodes) { };

    bool relevant(const ServerState &state, const TreeThinker & treeThinker) override {
        return true;
    }

    void exec(const ServerState & state, const TreeThinker & treeThinker) override {
        for (auto & child : children) {
            child.exec(state, treeThinker);
        }
    }
};



#endif //CSKNOW_NODES_H
