//
// Created by steam on 7/28/22.
//

#ifndef CSKNOW_CONDITION_HELPER_NODE_H
#define CSKNOW_CONDITION_HELPER_NODE_H

#include "node.h"

class TeamConditionDecorator : public ConditionDecorator {
    TeamId team;
public:
    TeamConditionDecorator(Blackboard & blackboard, Node::Ptr && node, TeamId team, string name = "TeamCondition") :
        ConditionDecorator(blackboard, std::move(node), (team == ENGINE_TEAM_T ? "T" : "CT") + name), team(team) { };
    virtual bool valid(const ServerState & state, TreeThinker & treeThinker) override {
        return state.getClient(treeThinker.csgoId).team == team;
    }
};

#endif //CSKNOW_CONDITION_HELPER_NODE_H
