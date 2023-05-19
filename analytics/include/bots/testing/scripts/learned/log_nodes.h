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
