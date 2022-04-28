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

    bool relevant(const ServerState &state) override {
        return true;
    }

    bool exec(const ServerState & state) override {
        child.exec(state);
    }

    void onEntry() override { }
    void onExit() override { }
};

class ParSelectorNode : Node {
    vector<Node> children;

public:
    ParSelectorNode(vector<Node> nodes) : children(nodes) { };

    bool relevant(const ServerState &state) override {
        return true;
    }

    bool exec(const ServerState & state) override {
        for (auto & child : children) {
            child.exec(state);
        }
    }

    void onEntry() override { }
    void onExit() override { }
};



#endif //CSKNOW_NODES_H
