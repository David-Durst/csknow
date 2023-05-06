//
// Created by durst on 5/6/23.
//

#include "bots/testing/scripts/test_setup.h"

bool SetupCommands::tick(ServerState &state, Blackboard & blackboard) {
    if (commands == nullptr) {
        commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                make_unique<SetBotStop>(blackboard, botStopStr),
                make_unique<SetMaxRounds>(blackboard, maxRounds, false)
        ), "Setup");
    }
    NodeState result = commands->exec(state, defaultThinker);
    return result == NodeState::Success;
}