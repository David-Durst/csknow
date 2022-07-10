//
// Created by durst on 7/7/22.
//

#include "queries/moments/push_vs_bait.h"

PushVsBaitResult queryPushVsBait(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick) {
    // for each round
        // for each player - identify current path - if in any regions of a path, or next region they will be in
        // for each time step in path - first player to shoot/be shot by enemy is pusher. baiter is everyone on same path
        // lurker is someone on different path
}
