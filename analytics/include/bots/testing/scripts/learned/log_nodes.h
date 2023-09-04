//
// Created by durst on 5/18/23.
//

#ifndef CSKNOW_LOG_NODES_H
#define CSKNOW_LOG_NODES_H

#include "bots/testing/script.h"
#include "bots/testing/command.h"
#include "bots/behavior_tree/tree.h"

namespace csknow::tests::learned {
    class StartNode : public SayCmd {
    public:
        StartNode(Blackboard & blackboard, const string & scriptName, size_t testIndex, size_t numTests) :
                SayCmd(blackboard, test_ready_string + "," + scriptName + "," + std::to_string(testIndex) +
                    "," + std::to_string(numTests), "StartNode") { }
    };

    class TraceStartNode : public SayCmd {
    public:
        TraceStartNode(Blackboard & blackboard, const string & scriptName, size_t roundIndex, size_t numRounds,
                       const string & demoFile, const size_t traceIndex, const size_t numTraces,
                       const string & nonReplayPlayers, bool oneTeam, bool oneBot) :
                SayCmd(blackboard, trace_ready_string + ":" + scriptName + ":" + std::to_string(roundIndex) +
                                   ":" + std::to_string(numRounds) + ":" + demoFile + ":" + std::to_string(traceIndex) +
                                   ":" + std::to_string(numTraces) + ":" + nonReplayPlayers + ":" +
                                   std::to_string(oneTeam) + ":" + std::to_string(oneBot), "StartNode") { }
    };

    class SuccessEndNode : public SayCmd {
    public:
        SuccessEndNode(Blackboard &blackboard, const string &scriptName, size_t testIndex, size_t numTests) :
                SayCmd(blackboard, test_finished_string + "," + scriptName + "," + std::to_string(testIndex) +
                                   "," + std::to_string(numTests), "SuccessEndNode") {};
    };

    class SayIfTimeoutEndNode : public SelectorNode {
    public:
        SayIfTimeoutEndNode(Blackboard & blackboard, const string & scriptName, size_t testIndex, size_t numTests,
                             double waitSeconds) :
                SelectorNode(blackboard, Node::makeList(
                        make_unique<movement::WaitNode>(blackboard, waitSeconds, false),
                        make_unique<SayCmd>(blackboard, test_failed_string + "," + scriptName + "," +
                                            std::to_string(testIndex) + "," + std::to_string(numTests))
                    ), "SayIfTimeoutEndNode") { };
    };

    class FailIfTimeoutEndNode : public SequenceNode {
    public:
        FailIfTimeoutEndNode(Blackboard & blackboard, const string & scriptName, size_t testIndex, size_t numTests,
                            double waitSeconds) :
                SequenceNode(blackboard, Node::makeList(
                        make_unique<SayIfTimeoutEndNode>(blackboard, scriptName, testIndex, numTests, waitSeconds),
                        make_unique<FailureNode>(blackboard)), "FailIfTimeoutEndNode") { };

    };
}

#endif //CSKNOW_LOG_NODES_H
