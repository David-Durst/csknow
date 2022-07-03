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
    Node::Ptr commands;

public:
    using Ptr = std::unique_ptr<Script>;
    string name;
    string curLog;

    Script(string name, vector<NeededBot> neededBots, ObserveSettings observeSettings) :
            name(name), neededBots(neededBots), observeSettings(observeSettings) { }
           //vector<Command::Ptr> && logicCommands, unique_ptr<Node> && conditions) :
           //logicCommands(std::move(logicCommands)), conditions(std::move(conditions)) { }

    TreeThinker getDefaultThinker(ServerState & state) {
        TreeThinker defaultThinker;
        defaultThinker.csgoId = state.clients[0].csgoId;
        return defaultThinker;
    }

    virtual void initialize(Tree & tree, ServerState & state);
    //virtual vector<string> generateCommands(ServerState & state);
    // prints result, returns when done
    bool tick(Tree & tree, ServerState & state);

    template <typename ...Args>
    static vector<Script::Ptr> makeList(Args ...args)
    {
        vector<Script::Ptr> scripts;
        constexpr size_t n = sizeof...(Args);
        scripts.reserve(n);

        (
                scripts.emplace_back(std::move(args)), ...
        );

        return scripts;
    }

    void restart(ServerState & state) { commands->restart(getDefaultThinker(state)); }
};

class ScriptsRunner {
protected:
    vector<Script::Ptr> scripts;
    size_t curScript = 0;
    bool startingNewScript = true;
    bool restartOnFinish;

public:
    ScriptsRunner(vector<Script::Ptr> && scripts, bool restartOnFinish = false) : scripts(std::move(scripts)), restartOnFinish(restartOnFinish) {
        if (this->scripts.empty()) {
            std::cout << "warning: scripts runner will crash with no scripts" << std::endl;
        }
    }

    void initialize(Tree & tree, ServerState & state);

    // return true when restarting
    bool tick(Tree & tree, ServerState & state);

    void restart(ServerState & state) {
        for (const auto & script : scripts) {
            script->restart(state);
        }
    }

    string curLog() { return scripts[curScript]->curLog; }
};

#endif //CSKNOW_SCRIPT_H
