//
// Created by durst on 12/25/22.
//

#include "bots/streaming_moments/streaming_fire_history.h"

namespace csknow::fire_history {
    void StreamingFireHistory::addTickData(const StreamingBotDatabase & db) {
        const ServerState & curState = db.batchData.fromNewest();
        set<CSGOId> activeClients;

        set<CSGOId> firingClients;
        for (const auto & weaponFireEvent : curState.weaponFireEvents) {
            firingClients.insert(weaponFireEvent.shooter);
        }

        map<CSGOId, set<CSGOId>> attackerToVictims;
        for (const auto & hurtEvent : curState.hurtEvents) {
            attackerToVictims[hurtEvent.attackerId].insert(hurtEvent.victimId);
        }

        for (const auto & curTickClient : curState.clients) {
            activeClients.insert(curTickClient.csgoId);
            fireClientHistory.addClient(curTickClient.csgoId);
            // if player is fresh or dead, give them a fresh state independent of past history
            FireClientData fireClientData;
            fireClientData.playerId = curTickClient.csgoId;
            if (db.clientHistoryLength(curTickClient.csgoId) < 2 || !curTickClient.isAlive) {
                fireClientData.holdingAttackButton = false;
                fireClientData.ticksSinceLastFire = MAX_TICKS_SINCE_LAST_FIRE_ATTACK;
                fireClientData.ticksSinceLastHoldingAttack = MAX_TICKS_SINCE_LAST_FIRE_ATTACK;
                fireClientData.hitEnemy = false;
                fireClientData.victims = {};
            }
            // otherwise, update based on cur tick/last tick
            else {
                const ServerState::Client & priorTickClient =
                    db.batchData.fromNewest(1).getClient(curTickClient.csgoId);
                const FireClientData & priorFireClientData =
                    fireClientHistory.clientHistory.at(curTickClient.csgoId).fromNewest();

                // if input angle defined, then action was input and firing is valid (if can fire)
                if (curTickClient.inputAngleDefined && curTickClient.intendedToFire &&
                    curTickClient.nextPrimaryAttack <= curState.gameTime) {
                    fireClientData.ticksSinceLastFire = 0;
                }
                /*
                if (firingClients.find(curTickClient.csgoId) != firingClients.end()) {
                    fireClientData.ticksSinceLastFire = 0;
                }
                 */
                else {
                    fireClientData.ticksSinceLastFire = std::min(MAX_TICKS_SINCE_LAST_FIRE_ATTACK,
                                                                 priorFireClientData.ticksSinceLastFire + 1);
                }

                // holding attack if not reloading and recoil index going up or holding constant and greater than 0.5
                // recoil index increases by 1.0 increments, so can't be 0.5 and increasing
                // (technically could check closer to 0, but why push limits?)
                // recoil index resets to 0 on weapon switch, so anything that isn't 0 and isnt reloading
                // results from firing
                // recoil index is constant in between shots, only goes down over time
                double curRecoilIndex = curTickClient.recoilIndex;
                double priorRecoilIndex = priorTickClient.recoilIndex;
                bool isReloading = curTickClient.isReloading;
                // recoil index holds constant on some frames when decaying
                // so need to make sure equality only counts for holding if haven't started decaying
                // circular holdingAttackButton works because all initialized to false and start on second tick of history
                bool recoilIndexNotDecaying = curRecoilIndex > priorRecoilIndex ||
                                              (curRecoilIndex == priorRecoilIndex &&
                                              priorFireClientData.holdingAttackButton);
                fireClientData.holdingAttackButton = !isReloading && curRecoilIndex > 0.5 && recoilIndexNotDecaying;
                // if fired last tick, make sure record attacking even though won't show up in data until next tick
                if (curTickClient.intendedToFire) { //fireClientData.holdingAttackButton || fireClientData.ticksSinceLastFire == 0) {
                    fireClientData.ticksSinceLastHoldingAttack = 0;
                }
                else {
                    fireClientData.ticksSinceLastHoldingAttack =
                        std::min(MAX_TICKS_SINCE_LAST_FIRE_ATTACK, priorFireClientData.ticksSinceLastHoldingAttack + 1);
                }

                if (attackerToVictims.find(curTickClient.csgoId) != attackerToVictims.end()){
                    fireClientData.hitEnemy = true;
                    fireClientData.victims = attackerToVictims[curTickClient.csgoId];
                }
                else {
                    fireClientData.hitEnemy = false;
                    fireClientData.victims = {};
                }
            }

            fireClientHistory.clientHistory.at(curTickClient.csgoId).enqueue(fireClientData);
        }

        fireClientHistory.removeInactiveClients(activeClients);
    }
}
