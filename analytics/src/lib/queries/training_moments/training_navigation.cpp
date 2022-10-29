//
// Created by durst on 10/20/22.
//

#include "queries/training_moments/training_navigation.h"
#include "bots/analysis/save_map_state.h"
#include "bots/analysis/vis_geometry.h"
#include "queries/lookback.h"
#include "queries/rolling_window.h"
#include <atomic>
#include "file_helpers.h"

namespace csknow::navigation {
    struct NavTrajData {
        int64_t roundId = INVALID_ID, trajectoryId = INVALID_ID, playerId = INVALID_ID;
        vector<int64_t> tickIds, patIds;
    };

    void recordSegments(vector<vector<int64_t>> &tmpRoundId,
                        vector<vector<int64_t>> &tmpTrajectoryId,
                        vector<vector<int64_t>> &tmpSegmentStartTickId,
                        vector<vector<int64_t>> &tmpSegmentCurTickId,
                        vector<vector<int64_t>> &tmpSegmentCurDemoTickId,
                        vector<vector<int64_t>> &tmpSegmentCurGameTickId,
                        vector<vector<int64_t>> &tmpSegmentFutureTickId,
                        vector<vector<array<int64_t, TOTAL_NAV_TICKS>>> &tmpSegmentTickIds,
                        vector<vector<array<int64_t, TOTAL_NAV_TICKS>>> &tmpSegmentPATIds,
                        vector<vector<int64_t>> &tmpLength, vector<vector<int64_t>> &tmpPlayerId,
                        vector<vector<TeamId>> &tmpTeamId,
                        vector<vector<array<Vec2, TOTAL_NAV_TICKS>>> &tmpPlayerViewDir,
                        vector<vector<array<double, TOTAL_NAV_TICKS>>> &tmpHealth,
                        vector<vector<array<double, TOTAL_NAV_TICKS>>> &tmpArmor,
                        vector<vector<MovementResult>> &tmpMovementResult,
                        int threadNum, const PlayerAtTick & playerAtTick,
                        const Ticks & ticks, const TickRates & tickRates,
                        const vector<NavTrajData> &finishedSegmentPerRound) {
        for (const auto &ntData: finishedSegmentPerRound) {
            // ensure enough past and future space by adding to start and removing end
            for (size_t ntTickNum = PAST_NAV_TICKS; ntTickNum < ntData.tickIds.size(); ntTickNum++) {
                // find the future tick id or break as finished creating training points from this trajectory
                int64_t futureTickId = INVALID_ID;
                int64_t futurePATId = INVALID_ID;
                for (size_t futureNTTickNum = ntTickNum + 1; futureNTTickNum < ntData.tickIds.size(); futureNTTickNum++) {
                    if (secondsBetweenTicks(ticks, tickRates, ntData.tickIds[ntTickNum],
                                            ntData.tickIds[futureNTTickNum]) >= FUTURE_NAV_SECONDS) {
                        futureTickId = ntData.tickIds[futureNTTickNum];
                        futurePATId = ntData.patIds[futureNTTickNum];
                        break;
                    }
                }
                if (futureTickId == INVALID_ID) {
                    break;
                }

                // handle everything thats individual ticks
                tmpRoundId[threadNum].push_back(ntData.roundId);
                tmpTrajectoryId[threadNum].push_back(ntData.trajectoryId);
                tmpSegmentStartTickId[threadNum].push_back(ntData.tickIds[ntTickNum - PAST_NAV_TICKS]);
                tmpSegmentCurTickId[threadNum].push_back(ntData.tickIds[ntTickNum]);
                tmpSegmentCurDemoTickId[threadNum].push_back(ticks.demoTickNumber[tmpSegmentCurTickId[threadNum].back()]);
                tmpSegmentCurGameTickId[threadNum].push_back(ticks.gameTickNumber[tmpSegmentCurTickId[threadNum].back()]);
                tmpSegmentFutureTickId[threadNum].push_back(futureTickId);
                tmpLength[threadNum].push_back(
                    tmpSegmentFutureTickId[threadNum].back() - tmpSegmentStartTickId[threadNum].back() + 1);
                tmpPlayerId[threadNum].push_back(ntData.playerId);

                // record the array of ticks
                tmpSegmentTickIds[threadNum].push_back({});
                tmpSegmentPATIds[threadNum].push_back({});
                size_t arrayId = 0;
                for (size_t pastNTTickNum = ntTickNum - PAST_NAV_TICKS; pastNTTickNum < ntTickNum; pastNTTickNum++) {
                    tmpSegmentTickIds[threadNum].back()[arrayId] = ntData.tickIds[pastNTTickNum];
                    tmpSegmentPATIds[threadNum].back()[arrayId] = ntData.patIds[pastNTTickNum];
                    arrayId++;
                }
                tmpSegmentTickIds[threadNum].back()[arrayId] = ntData.tickIds[ntTickNum];
                tmpSegmentPATIds[threadNum].back()[arrayId] = ntData.patIds[ntTickNum];
                tmpTeamId[threadNum].push_back(playerAtTick.team[ntData.patIds[ntTickNum]]);
                int64_t curPAT = tmpSegmentPATIds[threadNum].back()[arrayId];
                int64_t priorPAT = tmpSegmentPATIds[threadNum].back()[arrayId-1];
                arrayId++;
                tmpSegmentTickIds[threadNum].back()[arrayId] = futureTickId;
                tmpSegmentPATIds[threadNum].back()[arrayId] = futurePATId;

                // record the per tick data
                tmpPlayerViewDir[threadNum].push_back({});
                tmpHealth[threadNum].push_back({});
                tmpArmor[threadNum].push_back({});
                for (size_t i = 0; i < TOTAL_NAV_TICKS; i++) {
                    int64_t patId = tmpSegmentPATIds[threadNum].back()[i];
                    tmpPlayerViewDir[threadNum].back()[i] = {
                        playerAtTick.viewX[patId],
                        playerAtTick.viewY[patId]
                    };
                    tmpHealth[threadNum].back()[i] = playerAtTick.health[patId];
                    tmpArmor[threadNum].back()[i] = playerAtTick.armor[patId];
                }

                // record the result
                double xPosDelta = playerAtTick.posX[curPAT] - playerAtTick.posX[priorPAT];
                double yPosDelta = playerAtTick.posY[curPAT] - playerAtTick.posY[priorPAT];
                tmpMovementResult[threadNum].push_back({});
                if (std::abs(xPosDelta) < POS_DELTA_THRESHOLD) {
                    tmpMovementResult[threadNum].back().xResult = MovementBins::CONSTANT;
                }
                else if (xPosDelta < 0) {
                    tmpMovementResult[threadNum].back().xResult = MovementBins::DECREASE;
                }
                else {
                    tmpMovementResult[threadNum].back().xResult = MovementBins::INCREASE;
                }
                if (std::abs(yPosDelta) < POS_DELTA_THRESHOLD) {
                    tmpMovementResult[threadNum].back().yResult = MovementBins::CONSTANT;
                }
                else if (yPosDelta < 0) {
                    tmpMovementResult[threadNum].back().yResult = MovementBins::DECREASE;
                }
                else {
                    tmpMovementResult[threadNum].back().yResult = MovementBins::INCREASE;
                }
            }
        }
    }

    void createNavigationImages(const VisPoints &visPoints, const ReachableResult &reachableResult,
                                const Players &players, const Rounds &rounds,
                                const Ticks &ticks, const PlayerAtTick &playerAtTick,
                                const string &outputDir, const vector<set<int64_t>> & roundSyncTicks) {
        std::cout << "creating training nav images" << std::endl;
        std::atomic<int64_t> roundsProcessed = 0;

        CellBits bounds;
        for (const auto &cellVisPoint: visPoints.getCellVisPoints()) {
            bounds.set(cellVisPoint.cellId, true);
        }
        MapState boundsState(visPoints, bounds);
        MapState noBarrierState(visPoints);

        // for each round
        // for each tick
        // check when each player is in a region visible to enemy team
        //      save all alive player pos
        //      save what each alive player can see
        //      combine all alive players per team, save what each team can see
        //      blur each player if not seen since last sync, otherwise make them a point, then save || combo of enemies
        //      save distance map from each player pos to all other points
        //      for t's, save c4 pos. For ct's, save c4 pos if seen recently. otehrwise keep blurring and save
        //          future: handle bomb planted impact on pos knowledge
        //      only save state when on a sync clock for all players (if you started your trajectory off clock,
        //      tough luck, not doing anything until next period)
//#pragma omp parallel for
        for (int64_t roundIndex = 0; roundIndex < 2LL/*rounds.size*/; roundIndex++) {
            string roundOutputDir = outputDir + "_" + std::to_string(roundIndex);
            createAndEmptyDirectory(roundOutputDir);

            RollingWindow rollingWindow(rounds, ticks, playerAtTick);

            // images created every sync tick
            vector<map<int64_t, TemporalImageNames>> syncToImageNames;

            // data updated every tick
            map<int64_t, int64_t> lastTickPlayerVisByEnemies;
            map<int64_t, int64_t> lastTickPlayerVisFromByEnemies;
            map<int64_t, MapState> playerPosForEnemiesVis;
            map<int64_t, MapState> playerPosForEnemiesVisFrom;
            int64_t lastTickC4SeenByCT = INVALID_ID;
            MapState c4PosForCT(visPoints), c4PosForT(visPoints);

            for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
                 tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {
                /*
                if (tickIndex != 4209) {
                    continue;
                }
                std::cout << tickIndex << std::endl;
                if (ticks.demoTickNumber[tickIndex] != 5195) {
                    continue;
                }
                 */

                map<int64_t, int64_t> curPlayerToPAT = rollingWindow.getPATIdForPlayerId(tickIndex);

                // need to track these every tick
                map<int64_t, CellId> playerCellIds;
                map<int64_t, CellId> secondPlayerCellIds;
                map<int64_t, CellBits> playerPos;
                map<int64_t, CellBits> playerVis;
                map<int64_t, CellBits> playerVisFrom;
                CellBits ctPos, tPos;
                CellBits ctVis, tVis;
                CellBits ctVisFrom, tVisFrom;

                // check if sync tick
                bool syncTick = roundSyncTicks[roundIndex].find(tickIndex) != roundSyncTicks[roundIndex].end();
                if (syncTick) {
                    syncToImageNames.push_back({});
                }

                // need to remember one CT and one T players paths so can save team data
                const TemporalImageNames ctImgNames = TemporalImageNames(tickIndex, "", ENGINE_TEAM_CT,
                                                                         roundOutputDir);
                const TemporalImageNames tImgNames = TemporalImageNames(tickIndex, "", ENGINE_TEAM_T,
                                                                        roundOutputDir);

                // pass 1: compute everything that is only per player or per one team
                for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                     patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                    if (playerAtTick.isAlive[patIndex]) {
                        int64_t playerId = playerAtTick.playerId[patIndex];
                        TeamId teamId = playerAtTick.team[patIndex];

                        // compute player pos
                        vector<CellIdAndDistance> cellIdsByDistances = visPoints.getCellVisPointsByDistance({
                            playerAtTick.posX[patIndex],
                            playerAtTick.posY[patIndex],
                            playerAtTick.eyePosZ[patIndex]
                        });
                        const CellVisPoint & playerCellVisPoint =
                            visPoints.getCellVisPoints()[cellIdsByDistances[0].cellId];
                        playerCellIds[playerId] = playerCellVisPoint.cellId;
                        bool skipCrouchedNextCell = playerCellVisPoint.cellDiscreteCoordinates[0] ==
                            visPoints.getCellVisPoints()[cellIdsByDistances[1].cellId].cellDiscreteCoordinates[0] &&
                            playerCellVisPoint.cellDiscreteCoordinates[1] ==
                            visPoints.getCellVisPoints()[cellIdsByDistances[1].cellId].cellDiscreteCoordinates[1];
                        const CellVisPoint & secondPlayerCellVisPoint =
                            visPoints.getCellVisPoints()[cellIdsByDistances[skipCrouchedNextCell ? 2 : 1].cellId];
                        secondPlayerCellIds[playerId] = secondPlayerCellVisPoint.cellId;
                        CellBits localPos;
                        localPos.set(playerCellVisPoint.cellId, true);
                        localPos.set(secondPlayerCellVisPoint.cellId, true);
                        playerPos[playerId] = localPos;

                        // compute player vis
                        // might want to consider recoil here at some point, but such a minor factor, no in first
                        // pass
                        Vec2 playerViewAngle{
                            playerAtTick.viewX[patIndex],
                            playerAtTick.viewY[patIndex]
                        };
                        playerVis[playerId] =
                            getCellsInFOV(visPoints, playerCellVisPoint.topCenter, playerViewAngle);
                        playerVis[playerId] |=
                            getCellsInFOV(visPoints, secondPlayerCellVisPoint.topCenter, playerViewAngle);
                        playerVisFrom[playerId] = playerCellVisPoint.visibleFromCurPoint;
                        playerVisFrom[playerId] |= secondPlayerCellVisPoint.visibleFromCurPoint;
                        playerVis[playerId] &= playerVisFrom[playerId];

                        // add to team pos
                        if (teamId == ENGINE_TEAM_CT) {
                            ctPos |= localPos;
                        } else {
                            tPos |= localPos;
                        }

                        // add to team vis
                        if (teamId == ENGINE_TEAM_CT) {
                            ctVis |= playerVis[playerId];
                            ctVisFrom |= playerVisFrom[playerId];
                        } else {
                            tVis |= playerVis[playerId];
                            tVisFrom |= playerVisFrom[playerId];
                        }

                        MapState mapState(visPoints);
                        if (syncTick) {
                            TemporalImageNames imgNames = TemporalImageNames(tickIndex,
                                                                             players.name[players.idOffset + playerId],
                                                                             teamId, roundOutputDir);
                            syncToImageNames.back()[playerId] = imgNames;
                            mapState.saveNewMapState(playerPos[playerId], imgNames.playerPos);
                            mapState.saveNewMapState(playerVis[playerId], imgNames.playerVis);
                            mapState.saveNewMapState(playerVisFrom[playerId], imgNames.playerVisFrom);
                            mapState = reachableResult.scaledCellClosenessMatrix[playerCellVisPoint.cellId];
                            mapState.saveMapState( imgNames.distanceMap);

                            CellBits goalPos = localPos;
                            MapState goalState(visPoints, goalPos);
                            goalState.spread(boundsState, noBarrierState);
                            goalState.spread(boundsState, noBarrierState);
                            goalState.spread(boundsState, noBarrierState);
                            goalState.spread(boundsState, noBarrierState);
                            goalState.saveMapState(imgNames.goalPos);
                        }
                    }
                }

                // pass 2 for each player compute their individual data that needs other team data (aka if visible to other team)
                // vis data
                MapState ctVisToEnemies(visPoints), tVisToEnemies(visPoints);
                MapState ctVisMapState(visPoints, ctVis), tVisMapState(visPoints, tVis);
                // vis from data
                MapState ctVisFromEnemies(visPoints), tVisFromEnemies(visPoints);
                MapState ctVisFromMapState(visPoints, ctVisFrom), tVisFromMapState(visPoints, tVisFrom);
                for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                     patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                    if (playerAtTick.isAlive[patIndex]) {
                        int64_t playerId = playerAtTick.playerId[patIndex];
                        TeamId teamId = playerAtTick.team[patIndex];
                        // assume seen on first tick
                        // vis to logic
                        if (lastTickPlayerVisByEnemies.find(playerId) == lastTickPlayerVisByEnemies.end() ||
                            (teamId == ENGINE_TEAM_CT && (tVis[playerCellIds[playerId]] || tVis[secondPlayerCellIds[playerId]])) ||
                            (teamId == ENGINE_TEAM_T && (ctVis[playerCellIds[playerId]] || ctVis[secondPlayerCellIds[playerId]])) ||
                             tickIndex <= rounds.freezeTimeEnd[roundIndex]) {
                            lastTickPlayerVisByEnemies[playerId] = tickIndex;
                            if (playerPosForEnemiesVis.find(playerId) != playerPosForEnemiesVis.end()) {
                                playerPosForEnemiesVis.erase(playerId);
                            }
                            playerPosForEnemiesVis.insert({playerId, MapState(visPoints, playerPos[playerId])});
                        } else if (syncTick){
                            if (teamId == ENGINE_TEAM_CT) {
                                playerPosForEnemiesVis.at(playerId).spread(boundsState, tVisMapState);
                            }
                            else {
                                playerPosForEnemiesVis.at(playerId).spread(boundsState, ctVisMapState);
                            }
                        }
                        // same logic for vis from
                        if (lastTickPlayerVisFromByEnemies.find(playerId) == lastTickPlayerVisFromByEnemies.end() ||
                            (teamId == ENGINE_TEAM_CT && (tVisFrom[playerCellIds[playerId]] || tVisFrom[secondPlayerCellIds[playerId]])) ||
                            (teamId == ENGINE_TEAM_T && (ctVisFrom[playerCellIds[playerId]] || ctVisFrom[secondPlayerCellIds[playerId]])) ||
                            tickIndex <= rounds.freezeTimeEnd[roundIndex]) {
                            lastTickPlayerVisFromByEnemies[playerId] = tickIndex;
                            if (playerPosForEnemiesVisFrom.find(playerId) != playerPosForEnemiesVisFrom.end()) {
                                playerPosForEnemiesVisFrom.erase(playerId);
                            }
                            playerPosForEnemiesVisFrom.insert({playerId, MapState(visPoints, playerPos[playerId])});
                        } else if (syncTick){
                            if (teamId == ENGINE_TEAM_CT) {
                                playerPosForEnemiesVisFrom.at(playerId).spread(boundsState, tVisFromMapState);
                            }
                            else {
                                playerPosForEnemiesVisFrom.at(playerId).spread(boundsState, ctVisFromMapState);
                            }
                        }
                        if (teamId == ENGINE_TEAM_CT) {
                            ctVisToEnemies |= playerPosForEnemiesVis.at(playerId);
                            ctVisFromEnemies |= playerPosForEnemiesVisFrom.at(playerId);
                        } else {
                            tVisToEnemies |= playerPosForEnemiesVis.at(playerId);
                            tVisFromEnemies |= playerPosForEnemiesVisFrom.at(playerId);
                        }
                    }
                }

                // save the maps that combine both team data (aka visibility of enemies and c4 to current team)
                if (syncTick) {
                    MapState teamMapState(visPoints);
                    teamMapState.saveNewMapState(ctPos, ctImgNames.friendlyPos);
                    teamMapState.saveNewMapState(tPos, tImgNames.friendlyPos);
                    ctVisMapState.saveMapState(ctImgNames.friendlyVis);
                    tVisMapState.saveMapState(tImgNames.friendlyVis);
                    ctVisToEnemies.saveMapState(tImgNames.visEnemies);
                    tVisToEnemies.saveMapState(ctImgNames.visEnemies);
                    ctVisFromMapState.saveMapState(ctImgNames.friendlyVisFrom);
                    tVisFromMapState.saveMapState(tImgNames.friendlyVisFrom);
                    ctVisFromEnemies.saveMapState(tImgNames.visFromEnemies);
                    tVisFromEnemies.saveMapState(ctImgNames.visFromEnemies);
                }

                // c4 vis
                const CellVisPoint &c4CellVisPoint = visPoints.getCellVisPoints()[
                    visPoints.getCellVisPointsByDistance({
                        ticks.bombX[tickIndex],
                        ticks.bombY[tickIndex],
                        ticks.bombZ[tickIndex]
                    })[0].cellId];
                CellBits c4Pos;
                c4Pos.set(c4CellVisPoint.cellId, true);
                c4PosForT = c4Pos;
                if (lastTickC4SeenByCT == INVALID_ID || ctVis[c4CellVisPoint.cellId] ||
                    tickIndex <= rounds.freezeTimeEnd[roundIndex]) {
                    lastTickC4SeenByCT = tickIndex;
                    c4PosForCT = c4Pos;
                } else if (syncTick) {
                    c4PosForCT.spread(boundsState, ctVisMapState);
                }
                if (syncTick) {
                    c4PosForCT.saveMapState(ctImgNames.c4Pos);
                    c4PosForT.saveMapState(tImgNames.c4Pos);
                }

            }

            string subDirCommand = "mkdir " + roundOutputDir + "/trainNavData && echo " + roundOutputDir + "/*.png | xargs mv -t " + roundOutputDir + "/trainNavData/ --";
            if (system(subDirCommand.c_str()) != 0) {
                std::cerr << "create sub dir failed" << std::endl;
            }
            string removeOldTar = "rm -f " + roundOutputDir + ".tar";
            if (system(removeOldTar.c_str()) != 0) {
                std::cerr << "remove old tar failed" << std::endl;
            }
            string tarCommand = "tar -cf " + roundOutputDir + ".tar -C " + roundOutputDir + " trainNavData/";
            if (system(tarCommand.c_str()) != 0) {
                std::cerr << "tar failed" << std::endl;
            }
            string removeSubDirCommand = "echo " + roundOutputDir + "/trainNavData/* | xargs rm -- && rmdir " + roundOutputDir + "/trainNavData && rmdir " + roundOutputDir;
            if (system(removeSubDirCommand.c_str()) != 0) {
                std::cerr << "remove sub dir failed" << std::endl;
            }

            roundsProcessed++;
            printProgress(roundsProcessed, rounds.size);
        }

        string removeOldMainTarCommand = "rm -f " + outputDir + ".tar";
        if (system(removeOldMainTarCommand.c_str()) != 0) {
            std::cerr << "remove old main tar failed" << std::endl;
        }
        string makeMainTarCommand = "mv " + outputDir + "_0.tar " + outputDir + ".tar";
        if (system(makeMainTarCommand.c_str()) != 0) {
            std::cerr << "make main tar failed" << std::endl;
        }
        string combineTars = "tar --concatenate --file=" + outputDir + ".tar " + outputDir + "_*";
        if (system(combineTars.c_str()) != 0) {
            std::cerr << "combine tars failed" << std::endl;
        }
        string removeExtraTars = "rm " + outputDir + "_*";
        if (system(removeExtraTars.c_str()) != 0) {
            std::cerr << "remove extra tars failed" << std::endl;
        }
    }

    vector<set<int64_t>> createNavResult(const Games &games, const Rounds &rounds,
                                         const Ticks &ticks, const PlayerAtTick & playerAtTick,
                                         const NonEngagementTrajectoryResult & nonEngagementTrajectoryResult,
                                         TrainingNavigationResult & result) {
        int numThreads = omp_get_max_threads();
        vector<vector<int64_t>> tmpRoundIds(numThreads);
        vector<vector<int64_t>> tmpRoundStarts(numThreads);
        vector<vector<int64_t>> tmpRoundSizes(numThreads);
        vector<vector<int64_t>> tmpRoundIdPerTrajectory(numThreads);
        vector<vector<int64_t>> tmpTrajectoryId(numThreads);
        vector<vector<int64_t>> tmpSegmentStartTickId(numThreads);
        vector<vector<int64_t>> tmpSegmentCurTickId(numThreads);
        vector<vector<int64_t>> tmpSegmentCurDemoTickId(numThreads);
        vector<vector<int64_t>> tmpSegmentCurGameTickId(numThreads);
        vector<vector<int64_t>> tmpSegmentFutureTickId(numThreads);
        vector<vector<array<int64_t, TOTAL_NAV_TICKS>>> tmpSegmentTickIds(numThreads);
        vector<vector<array<int64_t, TOTAL_NAV_TICKS>>> tmpSegmentPATIds(numThreads);
        vector<vector<int64_t>> tmpTickLength(numThreads);
        vector<vector<int64_t>> tmpPlayerId(numThreads);
        vector<vector<TeamId>> tmpTeamId(numThreads);
        vector<vector<array<Vec2, TOTAL_NAV_TICKS>>> tmpPlayerViewDir(numThreads);
        vector<vector<array<double, TOTAL_NAV_TICKS>>> tmpHealth(numThreads);
        vector<vector<array<double, TOTAL_NAV_TICKS>>> tmpArmor(numThreads);
        vector<vector<MovementResult>> tmpMovementResult(numThreads);

        vector<set<int64_t>> roundSyncTicks(rounds.size, set<int64_t>{});

        std::cout << "creating training nav trajectories" << std::endl;
        std::atomic<int64_t> roundsProcessed = 0;

        // another function call below will handle creating images, this just has to make
        // the trajectories and record the sync ticks
        // for each round
        // for each tick
        // if a player is in trajectory, start a segment for them if no active segment
        // record all sync ticks for players in trajectories
        // save trajectries on when they end
        // clear out at end of round with early termination
//#pragma omp parallel for
        for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
            int threadNum = omp_get_thread_num();
            tmpRoundIds[threadNum].push_back(roundIndex);
            tmpRoundStarts[threadNum].push_back(static_cast<int64_t>(tmpSegmentStartTickId[threadNum].size()));

            TickRates tickRates = computeTickRates(games, rounds, roundIndex);

            // first compute sync ticks
            vector<int64_t> syncTickIds;
            int64_t lastSyncTickId = INVALID_ID;
            for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
                 tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {
                if (lastSyncTickId == INVALID_ID ||
                    secondsBetweenTicks(ticks, tickRates, lastSyncTickId, tickIndex) >=
                    PAST_NAV_TICKS_SECONDS_DELTA) {
                    lastSyncTickId = tickIndex;
                    syncTickIds.push_back(lastSyncTickId);
                }
            }
            roundSyncTicks[roundIndex] = set<int64_t>(syncTickIds.begin(), syncTickIds.end());

            map<int64_t, NavTrajData> playerToCurTrajectory;
            vector<NavTrajData> finishedSegmentPerRound;
            RollingWindow rollingWindow(rounds, ticks, playerAtTick);


            // then use sync ticks to compute trajectories
            for (size_t syncId = 0; syncId < syncTickIds.size(); syncId++) {
                int64_t tickIndex = syncTickIds[syncId];
                map<int64_t, int64_t> curPlayerToPAT = rollingWindow.getPATIdForPlayerId(tickIndex);

                // this tracks if a player is in a trajectory from non-engagemnet trajectory result
                // compared to internal playerToCurTrajectory/finishedSegmentPerRound to determine how to update internal
                // values
                set<int64_t> playerInTrajectory;

                for (const auto &[_0, _1, trajectoryIndex]:
                    nonEngagementTrajectoryResult.trajectoriesPerTick.intervalToEvent.findOverlapping(tickIndex,
                                                                                                      tickIndex)) {
                    int64_t curPlayerId = nonEngagementTrajectoryResult.playerId[trajectoryIndex];
                    playerInTrajectory.insert(curPlayerId);
                    if (playerToCurTrajectory.find(curPlayerId) == playerToCurTrajectory.end()) {
                        int64_t curPATId = curPlayerToPAT[curPlayerId];
                        // probably not necessary, but just be defensive
                        if (playerAtTick.isAlive[curPATId]) {
                            playerToCurTrajectory[curPlayerId] = {
                                roundIndex, trajectoryIndex, curPlayerId, {}, {}
                            };
                        }
                    }
                }

                // write if trajectory ended
                // no need to worry about players that disappear or are dead
                // as non_engagement_trajectory filters out those trajectories
                // add cur sync tick is not ended
                for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                     patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                    int64_t playerId = playerAtTick.playerId[patIndex];
                    if (playerToCurTrajectory.find(playerId) != playerToCurTrajectory.end()) {
                        // handle ended trajectory
                        if (playerInTrajectory.find(playerId) == playerInTrajectory.end()) {
                            finishedSegmentPerRound.push_back(playerToCurTrajectory[playerId]);
                            playerToCurTrajectory.erase(playerId);
                        } else {
                            playerToCurTrajectory[playerId].tickIds.push_back(tickIndex);
                            playerToCurTrajectory[playerId].patIds.push_back(patIndex);
                        }
                    }
                }
            }

            // finish all trajectories active at end of round
            for (const auto &[playerId, tData]: playerToCurTrajectory) {
                finishedSegmentPerRound.push_back(playerToCurTrajectory[playerId]);
                // no need to erase, this whole map will get dropped at end of round
            }
            std::sort(finishedSegmentPerRound.begin(), finishedSegmentPerRound.end(),
                      [](const NavTrajData & a, const NavTrajData & b) {
                          return a.trajectoryId < b.trajectoryId ||
                                 (a.trajectoryId == b.trajectoryId && a.tickIds[0] < b.tickIds[0]);
                      });

            recordSegments(tmpRoundIdPerTrajectory,
                           tmpTrajectoryId,
                           tmpSegmentStartTickId,
                           tmpSegmentCurTickId,
                           tmpSegmentCurDemoTickId,
                           tmpSegmentCurGameTickId,
                           tmpSegmentFutureTickId,
                           tmpSegmentTickIds, tmpSegmentPATIds,
                           tmpTickLength, tmpPlayerId,
                           tmpTeamId,
                           tmpPlayerViewDir,
                           tmpHealth,
                           tmpArmor,
                           tmpMovementResult,
                           threadNum, playerAtTick,
                           ticks, tickRates,
                           finishedSegmentPerRound);
            tmpRoundSizes[threadNum].push_back(static_cast<int64_t>(tmpSegmentStartTickId[threadNum].size()) -
                                               tmpRoundStarts[threadNum].back());
            roundsProcessed++;
            printProgress(roundsProcessed, rounds.size);
        }

        mergeThreadResults(numThreads, result.rowIndicesPerRound, tmpRoundIds, tmpRoundStarts, tmpRoundSizes,
                           result.segmentStartTickId, result.size,
                           [&](int64_t minThreadId, int64_t tmpRowId) {
                               result.roundId.push_back(tmpRoundIdPerTrajectory[minThreadId][tmpRowId]);
                               result.trajectoryId.push_back(tmpTrajectoryId[minThreadId][tmpRowId]);
                               result.segmentStartTickId.push_back(tmpSegmentStartTickId[minThreadId][tmpRowId]);
                               result.segmentCurTickId.push_back(tmpSegmentCurTickId[minThreadId][tmpRowId]);
                               result.segmentCurDemoTickId.push_back(tmpSegmentCurDemoTickId[minThreadId][tmpRowId]);
                               result.segmentCurGameTickId.push_back(tmpSegmentCurGameTickId[minThreadId][tmpRowId]);
                               result.segmentFutureTickId.push_back(tmpSegmentFutureTickId[minThreadId][tmpRowId]);
                               result.segmentTickIds.push_back(tmpSegmentTickIds[minThreadId][tmpRowId]);
                               result.segmentPATIds.push_back(tmpSegmentPATIds[minThreadId][tmpRowId]);
                               result.tickLength.push_back(tmpTickLength[minThreadId][tmpRowId]);
                               result.playerId.push_back(tmpPlayerId[minThreadId][tmpRowId]);
                               result.teamId.push_back(tmpTeamId[minThreadId][tmpRowId]);
                               result.playerViewDir.push_back(tmpPlayerViewDir[minThreadId][tmpRowId]);
                               result.health.push_back(tmpHealth[minThreadId][tmpRowId]);
                               result.armor.push_back(tmpArmor[minThreadId][tmpRowId]);
                               result.movementResult.push_back(tmpMovementResult[minThreadId][tmpRowId]);
                           });
        return roundSyncTicks;
    }

    TrainingNavigationResult queryTrainingNavigation(const VisPoints &visPoints, const ReachableResult &reachableResult,
                                                     const Players &players,
                                                     const Games &games, const Rounds &rounds,
                                                     const Ticks &ticks, const PlayerAtTick &playerAtTick,
                                                     const NonEngagementTrajectoryResult &nonEngagementTrajectoryResult,
                                                     const string &outputDir, bool createImages) {
        string trainNavDir = outputDir + "/trainNavData";

        TrainingNavigationResult result(players);
        vector<set<int64_t>> roundSyncTicks =
            createNavResult(games, rounds, ticks, playerAtTick, nonEngagementTrajectoryResult, result);
        // this has to save sync ticks
        // then during window, I can just check if i'm on a sync tick and skip otherwise
        // rolling window doesn't work here as that assumes adjacent ticks, i need spaces in between ticks
        // create navigation images will generate my ticks
        // i could do a rolling window where I know the window is larger than the sync ticks
        // then I iterate through window to check if matching a sync tick
        if (createImages) {
            createNavigationImages(visPoints, reachableResult, players, rounds, ticks,
                                   playerAtTick, trainNavDir, roundSyncTicks);
        }
        result.trainNavDir = trainNavDir;
        return result;
    }

    void testNavImages(const VisPoints &visPoints, const string &outputDir) {
        string basicTestNavDir = outputDir + "/basicTestNavData";

        // create a fresh directory to save to
        createAndEmptyDirectory(basicTestNavDir);

        MapState mapState(visPoints);
        mapState = visPoints.getCellVisPoints()[2418].visibleFromCurPoint;
        mapState.saveMapState(basicTestNavDir + "/visibleFromMid.png");
        CellBits all1s;
        for (const auto &cellVisPoint: visPoints.getCellVisPoints()) {
            all1s.set(cellVisPoint.cellId, true);
        }
        MapState all1sState(visPoints, all1s);
        all1sState.saveMapState(basicTestNavDir + "/wholeMap.png");
        CellBits viewAngle = getCellsInFOV(visPoints, visPoints.getCellVisPoints()[16133].topCenter, {0., 0.});
        mapState = viewAngle;
        mapState.saveMapState(basicTestNavDir + "/BSideTSpawnToASide.png");

        CellBits leftViewAngle = getCellsInFOV(visPoints, visPoints.getCellVisPoints()[2418].topCenter, {90., 0.});
        mapState = leftViewAngle;
        mapState.saveMapState(basicTestNavDir + "/midLeft.png");

        CellBits leftVisibleViewAngle = leftViewAngle;
        leftVisibleViewAngle &= visPoints.getCellVisPoints()[2418].visibleFromCurPoint;
        MapState leftVisibleState(visPoints, leftVisibleViewAngle);
        leftVisibleState.saveMapState(basicTestNavDir + "/midLeftVisible.png");

        CellBits upViewAngle = getCellsInFOV(visPoints, visPoints.getCellVisPoints()[2418].topCenter, {90., -32.});
        mapState = upViewAngle;
        mapState.saveMapState(basicTestNavDir + "/midLeftUp.png");

        Vec3 downPos = visPoints.getCellVisPoints()[16133].topCenter;
        downPos.z += 90;
        CellBits straightDownViewAngle = getCellsInFOV(visPoints, downPos, {90., 90.});
        mapState = straightDownViewAngle;
        mapState.saveMapState(basicTestNavDir + "/straightDownTestViewAngle.png");

        CellBits straightUpViewAngle = getCellsInFOV(visPoints, downPos, {90., -90.});
        mapState = straightUpViewAngle;
        mapState.saveMapState(basicTestNavDir + "/straightUpTestViewAngle.png");

        MapState emptyMapState(visPoints);
        CellBits onePoint;
        onePoint.set(16133, true);
        MapState noBarrierState(visPoints), barrierState(visPoints);
        noBarrierState = onePoint;
        barrierState = onePoint;
        int numSeconds = 0;
        int numMS = 0;
        noBarrierState.saveMapState(basicTestNavDir + "/nobarrier_spread0_000.png");
        barrierState.saveMapState(basicTestNavDir + "/barrier_spread0_000.png");
        for (size_t i = 0; i < 4*40; i++) {
            numMS++;
            numMS %= 4;
            if (numMS == 0) {
                numSeconds++;
            }
            noBarrierState.spread(all1sState, emptyMapState);
            barrierState.spread(all1sState, leftVisibleState);
            string msString;
            if (numMS == 0) {
                msString = "000";
            }
            else if (numMS == 1) {
                msString = "250";
            }
            else if (numMS == 2) {
                msString = "500";
            }
            else {
                msString = "750";
            }
            noBarrierState.saveMapState(basicTestNavDir + "/nobarrier_spread" + std::to_string(numSeconds) + "_" + msString + ".png");
            barrierState.saveMapState(basicTestNavDir + "/barrier_spread" + std::to_string(numSeconds) + "_" + msString + ".png");
        }
        std::cout << "finished test nav images" << std::endl;
    }
}