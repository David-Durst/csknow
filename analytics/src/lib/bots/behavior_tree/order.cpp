//
// Created by durst on 5/2/22.
//

#include "bots/behavior_tree/order.h"

bool D2OrderTaskNode::relevant(const ServerState &state, const TreeThinker &treeThinker) {
    return state.mapName == "de_dust2";
}

void D2OrderTaskNode::exec(const ServerState &state, const TreeThinker &treeThinker) {

}

bool GeneralOrderTaskNode::relevant(const ServerState &state, const TreeThinker &treeThinker) {
    return true;
}

void GeneralOrderTaskNode::exec(const ServerState &state, const TreeThinker &treeThinker) {

}
