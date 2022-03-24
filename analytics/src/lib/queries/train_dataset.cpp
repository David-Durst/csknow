//
// Created by durst on 2/21/22.
//
#include "queries/lookback.h"
#include "queries/train_dataset.h"
#include "geometryNavConversions.h"
#include "bots/thinker.h"
#include <utility>
#include <cassert>

vector<TrainDatasetResult::TimeStepState>
addStepStatesForTick(const Ticks & ticks, const PlayerAtTick & playerAtTick, const int64_t gameId, const int64_t roundId, const int64_t tickIndex,
                          const nav_mesh::nav_file & navFile, const TrainDatasetResult::TimeStepState defaultTimeStepState) {
    vector<TrainDatasetResult::TimeStepState> result;
    // default is CT friends, T enemy, will flip for each player
    TrainDatasetResult::TimeStepState timeStepStateCT = defaultTimeStepState;
    TrainDatasetResult::TimeStepState timeStepStateT = defaultTimeStepState;
    timeStepStateCT.gameId = gameId;
    timeStepStateT.gameId = gameId;
    timeStepStateCT.roundId = roundId;
    timeStepStateT.roundId = roundId;
    timeStepStateCT.tickId = tickIndex;
    timeStepStateT.tickId = tickIndex;
    map<int64_t, std::pair<uint32_t, int64_t>> playerIdToAABBAndPATId;

    for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
         patIndex != -1 && patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
        nav_mesh::vec3_t pos {static_cast<float>(playerAtTick.posX[patIndex]),
                              static_cast<float>(playerAtTick.posY[patIndex]),
                              static_cast<float>(playerAtTick.posZ[patIndex])};
        size_t navId = navFile.m_area_ids_to_indices.at(navFile.get_nearest_area_by_position(pos).get_id());
        playerIdToAABBAndPATId[playerAtTick.playerId[patIndex]] = {navId, patIndex};

        assert(navId < timeStepStateCT.navStates.size());
        assert(navId < timeStepStateT.navStates.size());

        if (playerAtTick.team[patIndex] == INTERNAL_TEAM_CT) {
            timeStepStateCT.navStates[navId].numFriends++;
            timeStepStateT.navStates[navId].numEnemies++;
        }
        else if (playerAtTick.team[patIndex] == INTERNAL_TEAM_T) {
            timeStepStateCT.navStates[navId].numEnemies++;
            timeStepStateT.navStates[navId].numFriends++;
        }
    }

    for (const auto & [playerId, aabbAndPATId] : playerIdToAABBAndPATId) {
        if (!playerAtTick.isAlive[aabbAndPATId.second] ||
            (playerAtTick.team[aabbAndPATId.second] != INTERNAL_TEAM_CT &&
                playerAtTick.team[aabbAndPATId.second] != INTERNAL_TEAM_T)) {
            continue;
        }

        TrainDatasetResult::TimeStepState timeStepStateForPlayer = playerAtTick.team[aabbAndPATId.second] == INTERNAL_TEAM_CT ?
                timeStepStateCT : timeStepStateT;
        timeStepStateForPlayer.curArea = aabbAndPATId.first;
        timeStepStateForPlayer.patId = aabbAndPATId.second;
        timeStepStateForPlayer.team = playerAtTick.team[aabbAndPATId.second];
        timeStepStateForPlayer.pos = {playerAtTick.posX[aabbAndPATId.second], playerAtTick.posY[aabbAndPATId.second],
                                      playerAtTick.posZ[aabbAndPATId.second]};
        result.push_back(timeStepStateForPlayer);
    }
    return result;
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
    vector<int64_t> numNavAreas(numThreads, 0);

//#pragma omp parallel for
    for (int64_t roundIndex = 0; roundIndex < std::min(5L, rounds.size); roundIndex++) {
        if (strcmp(games.mapName[rounds.gameId[roundIndex]], games.mapName[0]) != 0) {
            continue;
        }

        std::cout << "round index" << roundIndex << std::endl;

        int threadNum = omp_get_thread_num();
        TickRates tickRates = computeTickRates(games, rounds, roundIndex);
        int64_t lookbackGameTicks = DECISION_SECONDS * tickRates.gameTickRate;

        // initialize the default state for this round based on number of navmesh entries
        string mapName = games.mapName[rounds.gameId[roundIndex]];
        const nav_mesh::nav_file & navFile = mapNavs.at(mapName);
        numNavAreas[threadNum] = navFile.m_area_count;
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

            int64_t nextDemoTickId = tickIndex;
            int64_t curDemoTickId = tickIndex - getLookbackDemoTick(rounds, ticks, playerAtTick, tickIndex, tickRates, lookbackGameTicks, 1000);
            int64_t lastDemoTickId = curDemoTickId - 1;
            int64_t oldDemoTickId = tickIndex - getLookbackDemoTick(rounds, ticks, playerAtTick, tickIndex, tickRates, 2 * lookbackGameTicks, 1000);
            auto nextStepStates = addStepStatesForTick(ticks, playerAtTick, rounds.gameId[roundIndex], roundIndex, nextDemoTickId,
                                 navFile, defaultTimeStepState);
            auto curStepStates = addStepStatesForTick(ticks, playerAtTick, rounds.gameId[roundIndex], roundIndex, curDemoTickId,
                                 navFile, defaultTimeStepState);
            auto lastStepStates = addStepStatesForTick(ticks, playerAtTick, rounds.gameId[roundIndex], roundIndex, lastDemoTickId,
                                 navFile, defaultTimeStepState);
            auto oldStepStates = addStepStatesForTick(ticks, playerAtTick, rounds.gameId[roundIndex], roundIndex, oldDemoTickId,
                                 navFile, defaultTimeStepState);

            // players could leave in between frames, don't store any situations where all clients aren't same
            if (nextStepStates.size() == curStepStates.size() && nextStepStates.size() == lastStepStates.size() &&
                nextStepStates.size() == oldStepStates.size()) {
                bool samePlayersInSamePositions = true;
                for (size_t i = 0; i < nextStepStates.size(); i++) {
                    if (playerAtTick.playerId[nextStepStates[i].patId] != playerAtTick.playerId[curStepStates[i].patId] ||
                        playerAtTick.playerId[nextStepStates[i].patId] != playerAtTick.playerId[lastStepStates[i].patId] ||
                        playerAtTick.playerId[nextStepStates[i].patId] != playerAtTick.playerId[oldStepStates[i].patId]) {
                        samePlayersInSamePositions = false;
                        break;
                    }
                }

                if (samePlayersInSamePositions) {
                    tmpNextState[threadNum].insert(tmpNextState[threadNum].end(), nextStepStates.begin(), nextStepStates.end());
                    tmpCurState[threadNum].insert(tmpCurState[threadNum].end(), curStepStates.begin(), curStepStates.end());
                    tmpLastState[threadNum].insert(tmpLastState[threadNum].end(), lastStepStates.begin(), lastStepStates.end());
                    tmpOldState[threadNum].insert(tmpOldState[threadNum].end(), oldStepStates.begin(), oldStepStates.end());
                }
            }
        }

        for (int64_t planIndex = planStartIndex; planIndex < tmpCurState[threadNum].size(); planIndex++) {
            TrainDatasetResult::TimeStepPlan plan;


            plan.deltaX = tmpNextState[threadNum][planIndex].pos.x - tmpCurState[threadNum][planIndex].pos.x;
            plan.deltaY = tmpNextState[threadNum][planIndex].pos.y - tmpCurState[threadNum][planIndex].pos.y;
            double deltaZ = tmpNextState[threadNum][planIndex].pos.z - tmpCurState[threadNum][planIndex].pos.z;

            size_t curAreaId = tmpCurState[threadNum][planIndex].curArea,
                nextAreaId = tmpNextState[threadNum][planIndex].curArea;
            Vec3 curAreaCenter(vec3tConv(navFile.m_areas[curAreaId].get_center()));
            Vec2 movementDir{plan.deltaX, plan.deltaY},
                curAreaDir{curAreaCenter.x - tmpCurState[threadNum][planIndex].pos.x,
                           curAreaCenter.y - tmpCurState[threadNum][planIndex].pos.y};
            Ray movementRay(tmpCurState[threadNum][planIndex].pos, {movementDir.x, movementDir.y, deltaZ});
            if (curAreaId != nextAreaId) {
                plan.navTargetArea = nextAreaId;
            }
            else if (computeDistance(tmpCurState[threadNum][planIndex].pos, curAreaCenter) >= MIN_DISTANCE_TO_NAV_POINT &&
                cosineSimilarity(movementDir, curAreaDir) > COSINE_SIMILARITY_THRESHOLD) {
                plan.navTargetArea = curAreaId;
            }
            else {
                const nav_mesh::nav_area & startArea = navFile.m_areas[curAreaId];
                auto& areaConnections = startArea.get_connections( );
                // default to not hitting anything, so just stay in current area
                // this captures not moving
                uint32_t nearestArea = curAreaId;
                double nearestDistance = std::numeric_limits<double>::max();
                for ( auto& connection : areaConnections ) {
                    auto &connection_area = navFile.get_area_by_id(connection.id);
                    // create some z
                    AABB areaAABB{vec3tConv(connection_area.m_nw_corner), vec3tConv(connection_area.m_se_corner)};
                    // create some extra vertical space for too small AABB
                    areaAABB.min.z -= 5.;
                    areaAABB.max.z += 5.;
                    double newT0, newT1;
                    if(intersectP(areaAABB, movementRay, newT0, newT1) &&
                        newT0 * computeMagnitude(movementRay.dir) < nearestDistance) {
                        nearestArea = navFile.m_area_ids_to_indices.at(connection.id);
                        nearestDistance = newT0 * computeMagnitude(movementRay.dir);
                    }
                }
                plan.navTargetArea = nearestArea;
            }

            plan.shootDuringNextThink = playerAtTick.primaryBulletsClip[tmpNextState[threadNum][planIndex].patId] !=
                    playerAtTick.primaryBulletsClip[tmpCurState[threadNum][planIndex].patId];
            plan.crouchDuringNextThink = playerAtTick.isCrouching[tmpNextState[threadNum][planIndex].patId];

            tmpPlan[threadNum].push_back(plan);
        }
    }

    TrainDatasetResult result;
    result.numNavAreas = 0;
    for (int i = 0; i < numThreads; i++) {
        result.numNavAreas = std::max(result.numNavAreas, numNavAreas[i]);
        for (int j = 0; j < tmpCurState[i].size(); j++) {
            result.tickId.push_back(tmpCurState[i][j].tickId);
            result.roundId.push_back(tmpCurState[i][j].roundId);
            result.sourcePlayerId.push_back(playerAtTick.playerId[tmpCurState[i][j].patId]);
            result.sourcePlayerName.push_back(players.name[result.sourcePlayerId.back() + players.idOffset]);
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
