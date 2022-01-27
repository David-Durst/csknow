#include "bots/thinker.h"

void Thinker::think() {
    if (curBot >= state.clients.size()) {
        return;
    }
    int csknowId = state.serverClientIdToCSKnowId[curBot];
    state.inputsValid[csknowId] = true;
    state.clients[csknowId].buttons = 0;
    state.clients[csknowId].buttons |= IN_FORWARD;
    state.clients[csknowId].inputAngleDeltaPctY = 0.2;
}
