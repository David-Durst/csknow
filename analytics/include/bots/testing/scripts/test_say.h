//
// Created by durst on 9/3/23.
//

#ifndef CSKNOW_TEST_SAY_H
#define CSKNOW_TEST_SAY_H

#include "bots/testing/script.h"

class SayScript : public Script {
public:
    explicit SayScript(const ServerState &) :
            Script("SayScript", {{0, ENGINE_TEAM_CT}, {0, ENGINE_TEAM_CT}},
                   {ObserveType::Absolute, 0, {366.774475, 2669.538818, 239.860245}, {16.486465, -46.266056}}) { };

    void initialize(Tree & tree, ServerState & state) override {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            vector<Node::Ptr> sayCmds;
            for (size_t i = 0; i < 100; i++) {
                sayCmds.emplace_back(make_unique<SayCmd>(blackboard, std::to_string(i)));
            }
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         make_unique<InitTestingRound>(blackboard, name),
                                                         make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                                                         make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id, neededBots[1].id},state),
                                                         make_unique<SetPos>(blackboard, Vec3({1071.936035, 2972.308837, 128.762023}), Vec2({2.903987, -95.587982})),
                                                         make_unique<Teleport>(blackboard,neededBots[0].id, state),
                                                         make_unique<SetPos>(blackboard, Vec3({824.582764, 2612.630127, 95.957748}), Vec2({-1.760050, -105.049713})),
                                                         make_unique<Teleport>(blackboard, neededBots[1].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SequenceNode>(blackboard, std::move(sayCmds)),
                                                         make_unique<movement::WaitNode>(blackboard, 20, false)),
                                                 "SayScript");
        }
    }
};

#endif //CSKNOW_TEST_SAY_H
