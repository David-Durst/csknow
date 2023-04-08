//
// Created by durst on 1/5/23.
//

#include <bots/analysis/streaming_manager.h>
#include <bots/analysis/weapon_id_converter.h>
#include "bots/analysis/vis_geometry.h"

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

RoundPlantDefusal processRoundPlantDefusals(const Rounds & rounds, const Ticks & ticks, const Plants & plants,
                                            const Defusals & defusals, int64_t roundIndex) {
    RoundPlantDefusal result{INVALID_ID, INVALID_ID};
    for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
         tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {
        if (result.plantTickIndex == INVALID_ID) {
            for (const auto & [_0, _1, plantIndex] :
                ticks.plantsPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                if (plants.succesful[plantIndex]) {
                    result.plantTickIndex = plants.endTick[tickIndex];
                }
            }
        }
        if (result.defusalTickIndex == INVALID_ID) {
            for (const auto &[_0, _1, defusalIndex]:
                ticks.defusalsPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                if (defusals.succesful[defusalIndex]) {
                    result.defusalTickIndex = defusals.endTick[tickIndex];
                }
            }
        }
    }
    return result;
}

void StreamingManager::update(const Games & games, const RoundPlantDefusal & roundPlantDefusal, const Rounds & rounds,
                              const Players & players, const Ticks & ticks, const WeaponFire & weaponFire,
                              const Hurt & hurt, const PlayerAtTick & playerAtTick, int64_t tickIndex,
                              const csknow::nearest_nav_cell::NearestNavCell & nearestNavCell,
                              const VisPoints & visPoints, const TickRates & tickRates) {
    ServerState newState;

    int64_t roundIndex = ticks.roundId[tickIndex];
    int64_t gameIndex = rounds.gameId[roundIndex];

    newState.loadTime = secondsToCSKnowTime(gameTicksToSeconds(tickRates, ticks.gameTickNumber[tickIndex]));

    // loadGenerateState equivalent
    newState.mapName = games.mapName[gameIndex];
    newState.roundNumber = roundIndex;
    newState.tScore = rounds.tWins[gameIndex];
    newState.ctScore = rounds.ctWins[gameIndex];
    newState.mapNumber = gameIndex;
    newState.tickInterval = games.gameTickRate[gameIndex];
    newState.gameTime = ticks.gameTime[gameIndex];

    map<int64_t, int64_t> playerToPATindex;
    // loadClientStates equivelent
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
        playerToPATindex[playerAtTick.playerId[patIndex]] = patIndex;

        ServerState::Client newClient;
        newClient.lastFrame = static_cast<int32_t>(tickIndex);
        newClient.csgoId = static_cast<int32_t>(playerAtTick.playerId[patIndex]);
        newClient.lastTeleportId = INVALID_ID;
        newClient.name = players.name[playerAtTick.playerId[patIndex] + players.idOffset];
        newClient.team = static_cast<TeamId>(playerAtTick.team[patIndex]);
        newClient.health = static_cast<int>(playerAtTick.health[patIndex]);
        newClient.armor = static_cast<int>(playerAtTick.armor[patIndex]);
        newClient.hasHelmet = playerAtTick.hasHelmet[patIndex];
        newClient.currentWeaponId = static_cast<int32_t>(enumAsInt(
            demoEquipmentTypeToEngineWeaponId(playerAtTick.activeWeapon[patIndex])));
        newClient.nextPrimaryAttack = static_cast<float>(playerAtTick.nextPrimaryAttack[patIndex]);
        newClient.nextSecondaryAttack = static_cast<float>(playerAtTick.nextSecondaryAttack[patIndex]);
        newClient.timeWeaponIdle = INVALID_ID; // TODO: remove weapon idle, not useful
        newClient.recoilIndex = static_cast<float>(playerAtTick.recoilIndex[patIndex]);
        newClient.reloadVisuallyComplete = false; // TODO: remove reloading visually complete, not useful
        newClient.rifleId = static_cast<int32_t>(enumAsInt(
            demoEquipmentTypeToEngineWeaponId(playerAtTick.primaryWeapon[patIndex])));
        newClient.rifleClipAmmo = static_cast<int32_t>(playerAtTick.primaryBulletsClip[patIndex]);
        newClient.rifleReserveAmmo = static_cast<int32_t>(playerAtTick.primaryBulletsReserve[patIndex]);
        newClient.pistolId = static_cast<int32_t>(enumAsInt(
            demoEquipmentTypeToEngineWeaponId(playerAtTick.secondaryWeapon[patIndex])));
        newClient.pistolClipAmmo = static_cast<int32_t>(playerAtTick.secondaryBulletsClip[patIndex]);
        newClient.pistolReserveAmmo = static_cast<int32_t>(playerAtTick.secondaryBulletsReserve[patIndex]);
        newClient.flashes = static_cast<int32_t>(playerAtTick.numFlash[patIndex]);
        newClient.molotovs = static_cast<int32_t>(playerAtTick.numMolotov[patIndex]);
        newClient.smokes = static_cast<int32_t>(playerAtTick.numSmoke[patIndex]);
        newClient.hes = static_cast<int32_t>(playerAtTick.numHe[patIndex]);
        newClient.decoys = static_cast<int32_t>(playerAtTick.numDecoy[patIndex]);
        newClient.incendiaries = static_cast<int32_t>(playerAtTick.numIncendiary[patIndex]);
        newClient.zeus = static_cast<int32_t>(playerAtTick.numZeus[patIndex]);
        newClient.hasC4 = playerAtTick.hasBomb[patIndex];
        newClient.lastEyePosX = static_cast<float>(playerAtTick.posX[patIndex]);
        newClient.lastEyePosY = static_cast<float>(playerAtTick.posY[patIndex]);
        newClient.lastEyePosZ = static_cast<float>(playerAtTick.eyePosZ[patIndex]);
        newClient.lastFootPosZ = static_cast<float>(playerAtTick.posZ[patIndex]);
        newClient.lastVelX = static_cast<float>(playerAtTick.velX[patIndex]);
        newClient.lastVelY = static_cast<float>(playerAtTick.velY[patIndex]);
        newClient.lastVelZ = static_cast<float>(playerAtTick.velZ[patIndex]);
        newClient.lastEyeAngleX = static_cast<float>(playerAtTick.viewX[patIndex]);
        newClient.lastEyeAngleY = static_cast<float>(playerAtTick.viewY[patIndex]);
        newClient.lastAimpunchAngleX = static_cast<float>(playerAtTick.aimPunchX[patIndex]);
        newClient.lastAimpunchAngleY = static_cast<float>(playerAtTick.aimPunchY[patIndex]);
        newClient.lastViewpunchAngleX = static_cast<float>(playerAtTick.viewPunchX[patIndex]);
        newClient.lastViewpunchAngleY = static_cast<float>(playerAtTick.viewPunchY[patIndex]);
        newClient.lastEyeWithRecoilAngleX = static_cast<float>(viewWithRecoil.x);
        newClient.lastEyeWithRecoilAngleY = static_cast<float>(viewWithRecoil.y);
        newClient.isAlive = playerAtTick.isAlive[patIndex];
        newClient.isBot = true; // make all bots so tree looks at all
        newClient.isAirborne = playerAtTick.isAirborne[patIndex];
        newClient.isScoped = playerAtTick.isScoped[patIndex];
        newClient.duckAmount = static_cast<float>(playerAtTick.duckAmount[patIndex]);
        newClient.duckKeyPressed = playerAtTick.duckingKeyPressed[patIndex];
        newClient.isReloading = playerAtTick.isReloading[patIndex];
        newClient.isWalking = playerAtTick.isWalking[patIndex];
        newClient.flashDuration = static_cast<float>(playerAtTick.flashDuration[patIndex]);
        newClient.hasDefuser = playerAtTick.hasDefuser[patIndex];
        newClient.money = static_cast<int>(playerAtTick.money[patIndex]);
        newClient.ping = static_cast<int>(playerAtTick.ping[patIndex]);
        newClient.gameTime = static_cast<float>(playerAtTick.gameTime[patIndex]);

        newState.clients.push_back(newClient);
    }
    std::sort(newState.clients.begin(), newState.clients.end(),
              [](const ServerState::Client & a, const ServerState::Client & b) { return a.csgoId < b.csgoId; });

    // loadVisibilityClientPairs equivalent
    for (size_t outerClientIndex = 0; outerClientIndex < newState.clients.size(); outerClientIndex++) {
        const ServerState::Client & outerClient = newState.clients[outerClientIndex];
        for (size_t innerClientIndex = outerClientIndex + 1; innerClientIndex < newState.clients.size();
            innerClientIndex++) {
            const ServerState::Client & innerClient = newState.clients[innerClientIndex];
            if (demoIsVisible(playerAtTick, playerToPATindex[outerClient.csgoId],
                              playerToPATindex[innerClient.csgoId], nearestNavCell, visPoints)) {
                newState.visibilityClientPairs.insert({outerClient.csgoId, innerClient.csgoId});
            }
        }
    }

    // loadC4State equivalent
    newState.c4Exists = true;
    newState.c4IsPlanted = roundPlantDefusal.plantTickIndex != -1 && tickIndex >= roundPlantDefusal.plantTickIndex;
    newState.c4IsDropped = !newState.c4IsPlanted && ticks.bombCarrier[tickIndex] == INVALID_ID;
    newState.c4IsDefused = roundPlantDefusal.defusalTickIndex != -1 && tickIndex >= roundPlantDefusal.defusalTickIndex;
    newState.c4X = ticks.bombX[tickIndex];
    newState.c4Y = ticks.bombY[tickIndex];
    newState.c4Z = ticks.bombZ[tickIndex];

    // loadHurtEvent equivalent
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

    // loadWeaponFire equivalent
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

bool demoIsVisible(const PlayerAtTick & playerAtTick, int64_t attackerPATId, int64_t victimPATId,
                   const csknow::nearest_nav_cell::NearestNavCell & nearestNavCell,
                   const VisPoints & visPoints) {

    Vec3 attackerEyePos {
            playerAtTick.posX[attackerPATId],
            playerAtTick.posY[attackerPATId],
            playerAtTick.eyePosZ[attackerPATId]
    };

    Vec3 victimEyePos {
            playerAtTick.posX[victimPATId],
            playerAtTick.posY[victimPATId],
            playerAtTick.eyePosZ[victimPATId]
    };

    Vec2 curViewAngle {
            playerAtTick.viewX[attackerPATId],
            playerAtTick.viewY[attackerPATId]
    };
    vector<CellIdAndDistance> attackerCellIdsByDistances = nearestNavCell.getNearestCells(attackerEyePos);
    vector<CellIdAndDistance> victimCellIdsByDistances = nearestNavCell.getNearestCells(victimEyePos);
    /*
    vector<CellIdAndDistance> otherVictimCellIdsByDistances = visPoints.getCellVisPointsByDistance(
        victimEyePos);
    if (victimCellIdsByDistances[0].distance > otherVictimCellIdsByDistances[0].distance ||
        victimCellIdsByDistances[1].distance > otherVictimCellIdsByDistances[1].distance) {
        std::cout << "bad victim cell distance, pos: " << victimEyePos.toCSV() << std::endl;
    }
    */
    vector<CellVisPoint> victimTwoClosestCellVisPoints = {
            visPoints.getCellVisPoints()[victimCellIdsByDistances[0].cellId],
            visPoints.getCellVisPoints()[victimCellIdsByDistances[1].cellId]
    };
    bool victimInFOV = getCellsInFOV(victimTwoClosestCellVisPoints, attackerEyePos,
                                     curViewAngle);
    // vis from either of attackers two closest cell vis points
    bool victimVisNoFOV = false;
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            victimVisNoFOV |= visPoints.getCellVisPoints()[attackerCellIdsByDistances[i].cellId]
                    .visibleFromCurPoint[victimCellIdsByDistances[j].cellId];
        }
    }
    return victimInFOV && victimVisNoFOV;
}
