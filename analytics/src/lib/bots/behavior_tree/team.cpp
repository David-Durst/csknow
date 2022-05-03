//
// Created by durst on 4/26/22.
//

#include "bots/behavior_tree/node.h"

class PlantTaskNode : Node {
    double aSitePreference = 0.5;

    bool relevant(const ServerState & state, const TreeThinker & treeThinker) override {
        if (state.clients[treeThinker.curBotCSGOId].team != T_TEAM ||
            state.c4IsPlanted) {
            return false;
        }

        bool teamHasC4 = false;
        for (const auto & client : state.clients) {
            teamHasC4 |= client.hasC4;
        }
        if (teamHasC4) {

        }
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