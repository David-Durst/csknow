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
    bool initScript;

public:
    using Ptr = std::unique_ptr<Script>;
    string name;
    string curLog;

    Script(string name, vector<NeededBot> neededBots, ObserveSettings observeSettings, bool initScript = false) :
            name(name), neededBots(neededBots), observeSettings(observeSettings), initScript(initScript) { }
           //vector<Command::Ptr> && logicCommands, unique_ptr<Node> && conditions) :
           //logicCommands(std::move(logicCommands)), conditions(std::move(conditions)) { }

    TreeThinker getDefaultThinker(ServerState & state) {
        TreeThinker defaultThinker;
        defaultThinker.csgoId = INVALID_ID;
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

    const vector<NeededBot> & getNeededBots() { return neededBots; }
};

struct NeedPreTestingInitNode : Node {
    NeedPreTestingInitNode(Blackboard & blackboard) :
            Node(blackboard, "NeedPreTestingInitNode") { }
    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        blackboard.inTest = true;
        // need preinit testing if score is 0 0
        if (state.tScore == 0 && state.ctScore == 0) {
            playerNodeState[treeThinker.csgoId] = NodeState::Success;
            return playerNodeState[treeThinker.csgoId];
        }
        // need preinit testing if any humans aren't spectators
        for (const auto & client : state.clients) {
            if (!client.isBot && client.team != ENGINE_TEAM_SPEC) {
                playerNodeState[treeThinker.csgoId] = NodeState::Success;
                return playerNodeState[treeThinker.csgoId];
            }
        }
        // otherwise don't need score
        playerNodeState[treeThinker.csgoId] = NodeState::Failure;
        return playerNodeState[treeThinker.csgoId];
    }
};

class InitScript : public Script {
public:
    InitScript() : Script("InitScript", {}, {}, true) { };

    virtual void initialize(Tree & tree, ServerState & state) override  {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                    make_unique<NeedPreTestingInitNode>(blackboard),
                    make_unique<PreTestingInit>(blackboard),
                    make_unique<movement::WaitNode>(blackboard, 0.5)),
                 "InitScript");
        }
    }
};

class ScriptsRunner {
protected:
    vector<Script::Ptr> scripts;
    size_t curScript = 0;
    bool startingNewScript = true;
    bool restartOnFinish;

public:
    ScriptsRunner(vector<Script::Ptr> && scripts, bool restartOnFinish = false) : scripts(std::move(scripts)), restartOnFinish(restartOnFinish) {
        this->scripts.insert(this->scripts.begin(), make_unique<InitScript>());
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
