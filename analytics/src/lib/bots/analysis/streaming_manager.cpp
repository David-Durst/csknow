//
// Created by durst on 1/5/23.
//

#include <bots/analysis/streaming_manager.h>
#include <bots/analysis/weapon_id_converter.h>

void StreamingManager::update(const ServerState & state) {
    // if any client teleports, clear everyone's history
    for (const auto & client : state.clients) {
        if (client.lastTeleportId != client.lastTeleportConfirmationId || forceReset) {
            db.clear();
            streamingEngagementAim.reset();
            forceReset = false;
            streamingEngagementAim.aimTicksFile << "reset" << std::endl;
            break;
        }
    }

    streamingTestLogger.setCurFrameTime();
    if (streamingTestLogger.testActive()) {
        for (size_t i = 0; i < state.weaponFireEvents.size(); i++) {
            streamingTestLogger.addEvent("weapon fire", "");
        }
        for (size_t i = 0; i < state.hurtEvents.size(); i++) {
            streamingTestLogger.addEvent("hurt", "");
        }
    }

    db.addState(state);
    streamingFireHistory.addTickData(db);
    streamingEngagementAim.addTickData(db, streamingFireHistory);

    if (streamingTestLogger.testActive() && streamingTestLogger.attackerId != INVALID_ID &&
        state.getClient(streamingTestLogger.attackerId).isBot) {
        const EngagementAimTickData & attackerEngagementAimTickData =
            streamingEngagementAim.engagementAimPlayerHistory.clientHistory.at(streamingTestLogger.attackerId).fromNewest();
        streamingTestLogger.addEvent("angular distance x",
                                     std::to_string(attackerEngagementAimTickData.deltaRelativeCurHeadViewAngle.x));
        streamingTestLogger.addEvent("angular distance y",
                                     std::to_string(attackerEngagementAimTickData.deltaRelativeCurHeadViewAngle.y));
        streamingTestLogger.addEvent("angular target min x",
                                     std::to_string(attackerEngagementAimTickData.victimRelativeFirstHeadMinViewAngle.x));
        streamingTestLogger.addEvent("angular target min y",
                                     std::to_string(attackerEngagementAimTickData.victimRelativeFirstHeadMinViewAngle.y));
        streamingTestLogger.addEvent("angular target max x",
                                     std::to_string(attackerEngagementAimTickData.victimRelativeFirstHeadMaxViewAngle.x));
        streamingTestLogger.addEvent("angular target max y",
                                     std::to_string(attackerEngagementAimTickData.victimRelativeFirstHeadMaxViewAngle.y));
    }
}

void StreamingManager::update(const Players & players, const Ticks & ticks, const WeaponFire & weaponFire,
                              const Hurt & hurt, const PlayerAtTick & playerAtTick, int64_t tickIndex) {
    ServerState newState;

    for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
         patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
        Vec2 viewAngle {
            playerAtTick.viewX[patIndex],
            playerAtTick.viewY[patIndex]
        };

        Vec2 recoil {
            playerAtTick.aimPunchX[patIndex],
            playerAtTick.aimPunchY[patIndex]
        };

        Vec2 viewWithRecoil = viewAngle + recoil * WEAPON_RECOIL_SCALE;

        ServerState::Client newClient {
            static_cast<int32_t>(tickIndex),
            static_cast<int32_t>(playerAtTick.playerId[patIndex]),
            INVALID_ID,
            players.name[playerAtTick.playerId[patIndex]],
            static_cast<TeamId>(playerAtTick.team[patIndex]),
            static_cast<int>(playerAtTick.health[patIndex]),
            static_cast<int>(playerAtTick.armor[patIndex]),
            playerAtTick.hasHelmet[patIndex],
            static_cast<int32_t>(enumAsInt(
                demoEquipmentTypeToEngineWeaponId(playerAtTick.activeWeapon[patIndex]))),
            static_cast<float>(playerAtTick.nextPrimaryAttack[patIndex]),
            static_cast<float>(playerAtTick.nextSecondaryAttack[patIndex]),
            INVALID_ID, // TODO: remove weapon idle, not useful
            static_cast<float>(playerAtTick.recoilIndex[patIndex]),
            false, // TODO: remove reloading visually complete, not useful
            static_cast<int32_t>(enumAsInt(
                demoEquipmentTypeToEngineWeaponId(playerAtTick.primaryWeapon[patIndex]))),
            static_cast<int32_t>(playerAtTick.primaryBulletsClip[patIndex]),
            static_cast<int32_t>(playerAtTick.primaryBulletsReserve[patIndex]),
            static_cast<int32_t>(enumAsInt(
                demoEquipmentTypeToEngineWeaponId(playerAtTick.secondaryWeapon[patIndex]))),
            static_cast<int32_t>(playerAtTick.secondaryBulletsClip[patIndex]),
            static_cast<int32_t>(playerAtTick.secondaryBulletsReserve[patIndex]),
            static_cast<int32_t>(playerAtTick.numFlash[patIndex]),
            static_cast<int32_t>(playerAtTick.numMolotov[patIndex]),
            static_cast<int32_t>(playerAtTick.numSmoke[patIndex]),
            static_cast<int32_t>(playerAtTick.numHe[patIndex]),
            static_cast<int32_t>(playerAtTick.numDecoy[patIndex]),
            static_cast<int32_t>(playerAtTick.numIncendiary[patIndex]),
            static_cast<int32_t>(playerAtTick.numZeus[patIndex]),
            playerAtTick.hasBomb[patIndex],
            static_cast<float>(playerAtTick.posX[patIndex]),
            static_cast<float>(playerAtTick.posY[patIndex]),
            static_cast<float>(playerAtTick.eyePosZ[patIndex]),
            static_cast<float>(playerAtTick.posZ[patIndex]),
            static_cast<float>(playerAtTick.velX[patIndex]),
            static_cast<float>(playerAtTick.velY[patIndex]),
            static_cast<float>(playerAtTick.velZ[patIndex]),
            static_cast<float>(playerAtTick.viewX[patIndex]),
            static_cast<float>(playerAtTick.viewY[patIndex]),
            static_cast<float>(playerAtTick.aimPunchX[patIndex]),
            static_cast<float>(playerAtTick.aimPunchY[patIndex]),
            static_cast<float>(playerAtTick.viewPunchX[patIndex]),
            static_cast<float>(playerAtTick.viewPunchY[patIndex]),
            static_cast<float>(viewWithRecoil.x),
            static_cast<float>(viewWithRecoil.y),
            playerAtTick.isAlive[patIndex],
            true, // make all bots so tree looks at all
            playerAtTick.isAirborne[patIndex],
            playerAtTick.isScoped[patIndex],
            static_cast<float>(playerAtTick.duckAmount[patIndex]),
            playerAtTick.duckingKeyPressed[patIndex],
            playerAtTick.isReloading[patIndex],
            playerAtTick.isWalking[patIndex],
            static_cast<float>(playerAtTick.flashDuration[patIndex]),
            playerAtTick.hasDefuser[patIndex],
            static_cast<int>(playerAtTick.money[patIndex]),
            static_cast<int>(playerAtTick.ping[patIndex]),
            static_cast<float>(playerAtTick.gameTime[patIndex]),
        };
        newState.clients.push_back(newClient);
    }

    for (const auto & [_0, _1, hurtIndex] :
        ticks.hurtPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
        ServerState::Hurt newHurt {
            hurt.victim[hurtIndex],
            hurt.attacker[hurtIndex],
            hurt.health[hurtIndex],
            hurt.armor[hurtIndex],
            hurt.healthDamage[hurtIndex],
            hurt.armorDamage[hurtIndex],
            hurt.hitGroup[hurtIndex],
            demoEquipmentTypeToString(hurt.weapon[hurtIndex])
        };

        newState.hurtEvents.push_back(newHurt);
    }

    for (const auto & [_0, _1, weaponFireIndex] :
        ticks.weaponFirePerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
        ServerState::WeaponFire newWeaponFire {
            weaponFire.shooter[weaponFireIndex],
            demoEquipmentTypeToString(hurt.weapon[weaponFireIndex])
        };

        newState.weaponFireEvents.push_back(newWeaponFire);
    }


    newState.setClientIdTrackers();

    update(newState);
}
