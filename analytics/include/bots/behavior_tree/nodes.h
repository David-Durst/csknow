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

};

struct TreeThinker {
    // constant values across game
    int32_t curBotCSGOId;
    nav_mesh::nav_file navFile;
};

class Node {
    int64_t index;

public:
    virtual bool relevant(const ServerState & state, const TreeThinker & treeThinker);
    virtual bool exec(const ServerState & state, const TreeThinker & treeThinker);
    virtual void onEntry(const TreeThinker & treeThinker);
    virtual void onExit(const TreeThinker & treeThinker);
};

class RootNode : Node {
    Node child;

public:
    RootNode(Node node) : child(node) { };

    bool relevant(const ServerState &state, const TreeThinker & treeThinker) override {
        return true;
    }

    bool exec(const ServerState & state, const TreeThinker & treeThinker) override {
        child.exec(state, treeThinker);
    }

    void onEntry(const TreeThinker & treeThinker) override { }
    void onExit(const TreeThinker & treeThinker) override { }
};

class ParSelectorNode : Node {
    vector<Node> children;

public:
    ParSelectorNode(vector<Node> nodes) : children(nodes) { };

    bool relevant(const ServerState &state, const TreeThinker & treeThinker) override {
        return true;
    }

    bool exec(const ServerState & state, const TreeThinker & treeThinker) override {
        for (auto & child : children) {
            child.exec(state, treeThinker);
        }
    }

    void onEntry(const TreeThinker & treeThinker) override { }
    void onExit(const TreeThinker & treeThinker) override { }
};



#endif //CSKNOW_NODES_H
