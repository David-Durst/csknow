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
            neededBots(std::move(neededBots)), observeSettings(observeSettings),
            initScript(initScript), name(std::move(name)) { }
           //vector<Command::Ptr> && logicCommands, unique_ptr<Node> && conditions) :
           //logicCommands(std::move(logicCommands)), conditions(std::move(conditions)) { }

    static TreeThinker getDefaultThinker() {
        TreeThinker defaultThinker{INVALID_ID, AggressiveType::Push, {0., 0., 0., 0.}, 0.};
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

    void restart() { commands->restart(getDefaultThinker()); }

    const vector<NeededBot> & getNeededBots() { return neededBots; }
    vector<CSGOId> getNeededBotIds() {
        vector<CSGOId> result;
        for (const auto & neededBot : neededBots) {
            result.push_back(neededBot.id);
        }
        return result;
    }
};

struct NeedPreTestingInitNode : Node {
    int numHumansNonSpec;
    explicit NeedPreTestingInitNode(Blackboard & blackboard, int numHumansNonSpec) :
            Node(blackboard, "NeedPreTestingInitNode"), numHumansNonSpec(numHumansNonSpec) { }
    NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        blackboard.inTest = true;
        // need preinit testing if score is 0 0
        if (state.tScore == 0 && state.ctScore == 0) {
            playerNodeState[treeThinker.csgoId] = NodeState::Success;
            return playerNodeState[treeThinker.csgoId];
        }
        int numBots = 0;

        // need preinit testing if any humans aren't spectators
        int numHumans = 0;
        for (const auto & client : state.clients) {
            if (!client.isBot) {
                bool humanWrongTeam =
                    (client.team != ENGINE_TEAM_SPEC && numHumans >= numHumansNonSpec) ||
                    (client.team != ENGINE_TEAM_T && numHumans < numHumansNonSpec);
                numHumans++;
                if (humanWrongTeam) {
                    playerNodeState[treeThinker.csgoId] = NodeState::Success;
                    return playerNodeState[treeThinker.csgoId];
                }
            }
            if (client.isBot) {
                numBots++;
            }
        }
        // another human team check
        /*
        if (numBots < 7) {
            playerNodeState[treeThinker.csgoId] = NodeState::Success;
            return playerNodeState[treeThinker.csgoId];
        }
         */
        // otherwise don't need score
        playerNodeState[treeThinker.csgoId] = NodeState::Failure;
        return playerNodeState[treeThinker.csgoId];
    }
};

struct PreTestingInitFinishedNode : Node {
    int numHumansNonSpec;
    explicit PreTestingInitFinishedNode(Blackboard & blackboard, int numHumansNonSpec) :
            Node(blackboard, "PreTestingInitFinishedNode"), numHumansNonSpec(numHumansNonSpec) { }
    NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        blackboard.inTest = true;
        // need preinit testing if score is 0 0
        if (state.tScore == 0 && state.ctScore == 0) {
            playerNodeState[treeThinker.csgoId] = NodeState::Running;
            return playerNodeState[treeThinker.csgoId];
        }
        int numBots = 0;
        // need preinit testing if any humans aren't spectators
        int numHumans = 0;
        for (const auto & client : state.clients) {
            if (!client.isBot) {
                bool humanWrongTeam =
                    (client.team != ENGINE_TEAM_SPEC && numHumans >= numHumansNonSpec) ||
                    (client.team != ENGINE_TEAM_T && numHumans < numHumansNonSpec);
                numHumans++;
                if (humanWrongTeam) {
                    playerNodeState[treeThinker.csgoId] = NodeState::Running;
                    return playerNodeState[treeThinker.csgoId];
                }
            }
            if (client.isBot) {
                numBots++;
            }
        }
        // another human team check
        /*
         * this was a buggy check. human on spectator but only 6 bots as 7th hasn't spawned yet
        if (numBots < 7) {
            playerNodeState[treeThinker.csgoId] = NodeState::Running;
            return playerNodeState[treeThinker.csgoId];
        }
         */
        // otherwise don't need score
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
};

class InitScript : public Script {
    int numHumansNonSpec;
public:
    InitScript(int numHumansNonSpec) : Script("InitScript", {}, {}, true),
        numHumansNonSpec(numHumansNonSpec) { };

    void initialize(Tree & tree, ServerState & state) override  {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                    make_unique<NeedPreTestingInitNode>(blackboard, numHumansNonSpec),
                    make_unique<PreTestingInit>(blackboard, numHumansNonSpec),
                    make_unique<PreTestingInitFinishedNode>(blackboard, numHumansNonSpec),
                    make_unique<Draw>(blackboard),
                    make_unique<PreTestingInitFinishedNode>(blackboard, numHumansNonSpec)),
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
    explicit ScriptsRunner(vector<Script::Ptr> && scripts, bool restartOnFinish = false, int numHumans = 0) :
        scripts(std::move(scripts)), restartOnFinish(restartOnFinish) {
        // add Init as a separate script, so can skip it after it finishes
        this->scripts.insert(this->scripts.begin(), make_unique<InitScript>(numHumans));
        if (this->scripts.empty()) {
            std::cout << "warning: scripts runner will crash with no scripts" << std::endl;
        }
    }

    void initialize(Tree & tree, ServerState & state);

    // return true when restarting
    bool tick(Tree & tree, ServerState & state);

    void restart() {
        for (const auto & script : scripts) {
            script->restart();
        }
    }

    string curLog() { return scripts[curScript]->curLog; }
};

#endif //CSKNOW_SCRIPT_H
