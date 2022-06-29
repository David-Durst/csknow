//
// Created by steam on 6/19/22.
//

#ifndef CSKNOW_SCRIPT_H
#define CSKNOW_SCRIPT_H

#include "bots/behavior_tree/node.h"
#include "bots/testing/command.h"
#include "bots/testing/script_data.h"
#include "bots/behavior_tree/tree.h"
#include <memory>
using std::unique_ptr;

class Script {
protected:
    vector<NeededBot> neededBots;
    ObserveSettings observeSettings;
    string name;
    Node::Ptr commands;

public:
    string curLog;

    Script(string name, vector<NeededBot> neededBots, ObserveSettings observeSettings) :
            name(name), neededBots(neededBots), observeSettings(observeSettings) { }
           //vector<Command::Ptr> && logicCommands, unique_ptr<Node> && conditions) :
           //logicCommands(std::move(logicCommands)), conditions(std::move(conditions)) { }

    virtual void initialize(Tree & tree, ServerState & state);
    //virtual vector<string> generateCommands(ServerState & state);
    // prints result, returns when done
    bool tick(ServerState & state);
};

#endif //CSKNOW_SCRIPT_H
