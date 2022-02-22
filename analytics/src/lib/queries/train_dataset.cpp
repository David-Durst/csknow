//
// Created by durst on 2/21/22.
//
#include "queries/lookback.h"
#include "queries/train_dataset.h"
#include "geometryNavConversions.h"
#include <utility>

void addStepStatesForTick(const Ticks & ticks, const PlayerAtTick & playerAtTick, const int64_t gameId, const int64_t tickIndex,
                          const nav_mesh::nav_file & navFile, vector<TrainDatasetResult::TimeStepState> & stepStates,
                          const TrainDatasetResult::TimeStepState defaultTimeStepState) {
    // default is CT friends, T enemy, will flip for each player
    TrainDatasetResult::TimeStepState timeStepStateCT = defaultTimeStepState;
    TrainDatasetResult::TimeStepState timeStepStateT = defaultTimeStepState;
    timeStepStateCT.gameId = gameId;
    timeStepStateT.gameId = gameId;
    timeStepStateCT.tickId = tickIndex;
    timeStepStateT.tickId = tickIndex;
    map<int64_t, std::pair<uint32_t, int64_t>> playerIdToAABBAndPATId;

    for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
         patIndex != -1 && patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
        nav_mesh::vec3_t pos {static_cast<float>(playerAtTick.posX[patIndex]),
                              static_cast<float>(playerAtTick.posY[patIndex]),
                              static_cast<float>(playerAtTick.posZ[patIndex])};
        uint32_t navId = navFile.get_nearest_area_by_position(pos).get_id();
        playerIdToAABBAndPATId[playerAtTick.playerId[patIndex]] = {navId, patIndex};

        if (playerAtTick.team[patIndex] == TEAM_CT) {
            timeStepStateCT.navStates[navId].numFriends++;
            timeStepStateT.navStates[navId].numEnemies++;
        }
        else if (playerAtTick.team[patIndex] == TEAM_T) {
            timeStepStateCT.navStates[navId].numEnemies++;
            timeStepStateT.navStates[navId].numFriends++;
        }
    }

    for (const auto & [playerId, aabbAndPATId] : playerIdToAABBAndPATId) {
        if (!playerAtTick.isAlive[aabbAndPATId.second] ||
            (playerAtTick.team[aabbAndPATId.second] != TEAM_CT && playerAtTick.team[aabbAndPATId.second] != TEAM_T)) {
            continue;
        }

        TrainDatasetResult::TimeStepState timeStepStateForPlayer = playerAtTick.team[aabbAndPATId.second] == TEAM_CT ?
                timeStepStateCT : timeStepStateT;
        timeStepStateForPlayer.curAABB = aabbAndPATId.first;
        timeStepStateForPlayer.patId = aabbAndPATId.second;
        timeStepStateForPlayer.pos = {playerAtTick.posX[aabbAndPATId.second], playerAtTick.posY[aabbAndPATId.second],
                                      playerAtTick.posZ[aabbAndPATId.second]};
        stepStates.push_back(timeStepStateForPlayer);
    }
}


TrainDatasetResult queryTrainDataset(const Games & games, const Rounds & rounds, const Ticks & ticks,
                                     const Players & players, const PlayerAtTick & playerAtTick,
                                     const std::map<std::string, const nav_mesh::nav_file> & mapNavs) {

    int numThreads = omp_get_max_threads();
    vector<TrainDatasetResult::TimeStepState> tmpCurState[numThreads];
    vector<TrainDatasetResult::TimeStepState> tmpNextState[numThreads];
    vector<TrainDatasetResult::TimeStepState> tmpLastState[numThreads];
    vector<TrainDatasetResult::TimeStepState> tmpOldState[numThreads];
    vector<TrainDatasetResult::TimeStepPlan> tmpPlan[numThreads];

#pragma omp parallel for
    for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
        int threadNum = omp_get_thread_num();
        TickRates tickRates = computeTickRates(games, rounds, roundIndex);
        int64_t lookbackGameTicks = DECISION_SECONDS * tickRates.gameTickRate;

        // initialize the default state for this round based on number of navmesh entries
        string mapName = games.mapName[rounds.gameId[roundIndex]];
        const nav_mesh::nav_file & navFile = mapNavs.at(mapName);
        TrainDatasetResult::TimeStepState defaultTimeStepState(navFile.m_area_count);

        // remember start in tmp vectors for plan recording
        int64_t planStartIndex = tmpCurState[threadNum].size();

        int64_t lastTickInDataset = rounds.ticksPerRound[roundIndex].minId;
        for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
             tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {

            // only store every DECISION_SECONDS, not continually making a decision
            // make sure at least 2 DECISION_SECONDS from start so can look into future
            if (secondsBetweenTicks(ticks, tickRates, lastTickInDataset, tickIndex) < DECISION_SECONDS ||
                    secondsBetweenTicks(ticks, tickRates, rounds.ticksPerRound[roundIndex].minId, tickIndex) < 2*DECISION_SECONDS) {
                continue;
            }
            lastTickInDataset = tickIndex;

            int64_t nextDemoTickId = tickIndex - 1;
            int64_t lastDemoTickId = tickIndex - 1;
            int64_t oldDemoTickId = getLookbackDemoTick(rounds, ticks, playerAtTick, tickIndex, tickRates, lookbackGameTicks);
            addStepStatesForTick(ticks, playerAtTick, rounds.gameId[roundIndex], tickIndex,
                                 navFile, tmpCurState[threadNum], defaultTimeStepState);
            addStepStatesForTick(ticks, playerAtTick, rounds.gameId[roundIndex], nextDemoTickId,
                                 navFile, tmpNextState[threadNum], defaultTimeStepState);
            addStepStatesForTick(ticks, playerAtTick, rounds.gameId[roundIndex], lastDemoTickId,
                                 navFile, tmpLastState[threadNum], defaultTimeStepState);
            addStepStatesForTick(ticks, playerAtTick, rounds.gameId[roundIndex], oldDemoTickId,
                                 navFile, tmpOldState[threadNum], defaultTimeStepState);
        }

        for (int64_t planIndex = planStartIndex; planIndex < tmpCurState[threadNum].size(); planIndex++) {
            TrainDatasetResult::TimeStepPlan plan;

            plan.deltaX = tmpNextState[threadNum][planIndex].pos.x - tmpCurState[threadNum][planIndex].pos.x;
            plan.deltaY = tmpNextState[threadNum][planIndex].pos.y - tmpCurState[threadNum][planIndex].pos.y;

            plan.shootDuringNextThink = playerAtTick.primaryBulletsClip[tmpNextState[threadNum][planIndex].patId] !=
                    playerAtTick.primaryBulletsClip[tmpCurState[threadNum][planIndex].patId];
            plan.crouchDuringNextThink = playerAtTick.isCrouching[tmpNextState[threadNum][planIndex].patId];
        }
    }

    TrainDatasetResult result;
    for (int i = 0; i < numThreads; i++) {
        for (int j = 0; j < tmpCurState[i].size(); j++) {
            result.tickId.push_back(tmpCurState[i][j].tickId);
            result.sourcePlayerId.push_back(playerAtTick.playerId[tmpCurState[i][j].patId]);
            result.sourcePlayerName.push_back(players.name[result.sourcePlayerId.back()]);
            result.demoName.push_back(games.demoFile[tmpCurState[i][j].gameId]);
            result.curState.push_back(tmpCurState[i][j]);
            result.lastState.push_back(tmpLastState[i][j]);
            result.oldState.push_back(tmpOldState[i][j]);
            result.plan.push_back(tmpPlan[i][j]);
        }
    }
    result.size = result.tickId.size();
    return result;
}
