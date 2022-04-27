//
// Created by durst on 4/26/22.
//

#include "bots/behavior_tree/nodes.h"

class PlantTaskNode : Node {
    double aSitePreference = 0.5;

    bool relevant(const ServerState & state, int32_t curBotCSGOId) override {
        if (state.clients[curBotCSGOId].team == T_TEAM && )
    }
};

class EliminateTaskNode : Node {

};

class GuardTaskNode : Node {

};

class EscapeTaskNode : Node {

};

RootNode buildTeamTree() {

}