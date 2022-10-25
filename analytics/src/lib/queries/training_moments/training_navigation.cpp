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
        int64_t trajectoryId = INVALID_ID, playerId = INVALID_ID;
        vector<int64_t> tickIds, patIds;
    };

    void recordSegments(vector<vector<int64_t>> &tmpTrajectoryId,
                        vector<vector<int64_t>> &tmpSegmentStartTickId,
                        vector<vector<int64_t>> &tmpSegmentCurTickId,
                        vector<vector<int64_t>> &tmpSegmentFutureTickId,
                        vector<vector<array<int64_t, TOTAL_NAV_TICKS>>> &tmpSegmentTickIds,
                        vector<vector<array<int64_t, TOTAL_NAV_TICKS>>> &tmpSegmentPATIds,
                        vector<vector<int64_t>> &tmpLength, vector<vector<int64_t>> &tmpPlayerId,
                        vector<vector<TeamId>> &tmpTeamId,
                        vector<vector<array<Vec2, TOTAL_NAV_TICKS>>> &tmpPlayerViewDir,
                        vector<vector<array<double, TOTAL_NAV_TICKS>>> &tmpHealth,
                        vector<vector<array<double, TOTAL_NAV_TICKS>>> &tmpArmor,
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
                    }
                }
                if (futureTickId == INVALID_ID) {
                    break;
                }

                // handle everything thats individual ticks
                tmpTrajectoryId[threadNum].push_back(ntData.trajectoryId);
                tmpSegmentStartTickId[threadNum].push_back(ntData.tickIds[ntTickNum - PAST_NAV_TICKS]);
                tmpSegmentCurTickId[threadNum].push_back(ntData.tickIds[ntTickNum]);
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
            }
        }
    }

    void createNavigationImages(const VisPoints &visPoints, const ReachableResult &reachableResult,
                                const Players &players, const Rounds &rounds,
                                const Ticks &ticks, const PlayerAtTick &playerAtTick,
                                const string &outputDir, const vector<set<int64_t>> & roundSyncTicks) {
        std::cout << "creating training nav images" << std::endl;
        std::atomic<int64_t> roundsProcessed = 0;
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
#pragma omp parallel for
        for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
            RollingWindow rollingWindow(rounds, ticks, playerAtTick);

            // images created every sync tick
            vector<map<int64_t, TemporalImageNames>> syncToImageNames;

            // data updated every tick
            map<int64_t, int64_t> lastTickPlayerSeenByEnemies;
            map<int64_t, MapState> playerPosForEnemies;
            int64_t lastTickC4SeenByCT = INVALID_ID;
            MapState c4PosForCT(visPoints), c4PosForT(visPoints);

            for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
                 tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {
                map<int64_t, int64_t> curPlayerToPAT = rollingWindow.getPATIdForPlayerId(tickIndex);

                // need to track these every tick
                map<int64_t, CellId> playerCellIds;
                map<int64_t, CellBits> playerPos;
                map<int64_t, CellBits> playerVis;
                CellBits ctPos, tPos;
                CellBits ctVis, tVis;

                // check if sync tick
                bool syncTick = roundSyncTicks[roundIndex].find(tickIndex) != roundSyncTicks[roundIndex].end();
                if (syncTick) {
                    syncToImageNames.push_back({});
                }

                // need to remember one CT and one T players paths so can save team data
                const TemporalImageNames ctImgNames = TemporalImageNames(tickIndex, "", ENGINE_TEAM_CT,
                                                                         outputDir);
                const TemporalImageNames tImgNames = TemporalImageNames(tickIndex, "", ENGINE_TEAM_T,
                                                                        outputDir);

                // pass 1: compute everything that is only per player or per one team
                for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                     patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                    if (patIndex == 38175) {
                        int x = 1;
                        (void) x;
                    }
                    if (playerAtTick.isAlive[patIndex]) {
                        int64_t playerId = playerAtTick.playerId[patIndex];
                        TeamId teamId = playerAtTick.team[patIndex];

                        // compute player pos
                        const CellVisPoint &playerCellVisPoint = visPoints.getNearestCellVisPoint({
                            playerAtTick.posX[patIndex],
                            playerAtTick.posY[patIndex],
                            playerAtTick.eyePosZ[patIndex]
                        });
                        playerCellIds[playerId] = playerCellVisPoint.cellId;
                        CellBits localPos;
                        localPos.set(playerCellVisPoint.cellId, true);
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
                        playerVis[playerId] &= playerCellVisPoint.visibleFromCurPoint;

                        // add to team pos
                        if (teamId == ENGINE_TEAM_CT) {
                            ctPos |= localPos;
                        } else {
                            tPos |= localPos;
                        }

                        // add to team vis
                        if (teamId == ENGINE_TEAM_CT) {
                            ctVis |= playerVis[playerId];
                        } else {
                            tVis |= playerVis[playerId];
                        }

                        MapState mapState(visPoints);
                        if (syncTick) {
                            TemporalImageNames imgNames = TemporalImageNames(tickIndex,
                                                                             players.name[players.idOffset + playerId],
                                                                             teamId, outputDir);
                            syncToImageNames.back()[playerId] = imgNames;
                            mapState.saveNewMapState(playerPos[playerId], imgNames.playerPos);
                            mapState.saveNewMapState(playerVis[playerId], imgNames.playerVis);
                            mapState.saveNewMapState(
                                reachableResult.scaledCellDistanceMatrix[playerCellVisPoint.cellId],
                                imgNames.distanceMap);

                            CellBits goalPos;
                            const AreaVisPoint & playerAreaVisPoint = visPoints.getAreaVisPoint(playerCellVisPoint.areaId);
                            for (const auto & areaId : visPoints.getPlacesToAreas().at(playerAreaVisPoint.placeIndex)) {
                                const AreaVisPoint & placeAreaVisPoint = visPoints.getAreaVisPoint(areaId);
                                for (const auto & cellId : placeAreaVisPoint.cells) {
                                    goalPos.set(cellId, true);
                                }
                            }
                            mapState.saveNewMapState(goalPos, imgNames.goalPos);
                        }
                    }
                }

                // pass 2 for each player compute their individual data that needs other team data (aka if visible to other team)
                MapState ctVisToEnemies(visPoints), tVisToEnemies(visPoints);
                for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                     patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                    if (playerAtTick.isAlive[patIndex]) {
                        int64_t playerId = playerAtTick.playerId[patIndex];
                        TeamId teamId = playerAtTick.team[patIndex];
                        // assume seen on first tick
                        if (lastTickPlayerSeenByEnemies.find(playerId) == lastTickPlayerSeenByEnemies.end() ||
                            (teamId == ENGINE_TEAM_CT && tVis[playerCellIds[playerId]]) ||
                            (teamId == ENGINE_TEAM_T && ctVis[playerCellIds[playerId]])) {
                            lastTickPlayerSeenByEnemies[playerId] = tickIndex;
                            MapState posStateForEnemies(visPoints);
                            posStateForEnemies = playerPos[playerId];
                            playerPosForEnemies.insert({playerId, posStateForEnemies});
                        } else {
                            playerPosForEnemies.at(playerId).conv(UNIFORM_BLUR_MATRIX);
                        }
                        if (teamId == ENGINE_TEAM_CT) {
                            ctVisToEnemies |= playerPosForEnemies.at(playerId);
                        } else {
                            tVisToEnemies |= playerPosForEnemies.at(playerId);
                        }
                    }
                }

                // save the maps that combine both team data (aka visibility of enemies and c4 to current team)
                if (syncTick) {
                    MapState teamMapState(visPoints);
                    teamMapState.saveNewMapState(ctPos, ctImgNames.friendlyPos);
                    teamMapState.saveNewMapState(tPos, tImgNames.friendlyPos);
                    teamMapState.saveNewMapState(ctVis, ctImgNames.friendlyVis);
                    teamMapState.saveNewMapState(tVis, tImgNames.friendlyVis);
                    ctVisToEnemies.saveMapState(tImgNames.visEnemies);
                    tVisToEnemies.saveMapState(ctImgNames.visEnemies);
                }

                // c4 vis
                const CellVisPoint &c4CellVisPoint = visPoints.getNearestCellVisPoint({
                                                                                          ticks.bombX[tickIndex],
                                                                                          ticks.bombY[tickIndex],
                                                                                          ticks.bombZ[tickIndex]
                                                                                      });
                CellBits c4Pos;
                c4Pos.set(c4CellVisPoint.cellId, true);
                c4PosForT = c4Pos;
                if (lastTickC4SeenByCT == INVALID_ID || ctVis[c4CellVisPoint.cellId]) {
                    lastTickC4SeenByCT = tickIndex;
                    c4PosForCT = c4Pos;
                } else {
                    c4PosForCT.conv(UNIFORM_BLUR_MATRIX);
                }
                if (syncTick) {
                    c4PosForCT.saveMapState(ctImgNames.c4Pos);
                    c4PosForT.saveMapState(tImgNames.c4Pos);
                }

            }

            roundsProcessed++;
            printProgress(roundsProcessed, rounds.size);
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
        vector<vector<int64_t>> tmpTrajectoryId(numThreads);
        vector<vector<int64_t>> tmpSegmentStartTickId(numThreads);
        vector<vector<int64_t>> tmpSegmentCurTickId(numThreads);
        vector<vector<int64_t>> tmpSegmentFutureTickId(numThreads);
        vector<vector<array<int64_t, TOTAL_NAV_TICKS>>> tmpSegmentTickIds(numThreads);
        vector<vector<array<int64_t, TOTAL_NAV_TICKS>>> tmpSegmentPATIds(numThreads);
        vector<vector<int64_t>> tmpTickLength(numThreads);
        vector<vector<int64_t>> tmpPlayerId(numThreads);
        vector<vector<TeamId>> tmpTeamId(numThreads);
        vector<vector<array<Vec2, TOTAL_NAV_TICKS>>> tmpPlayerViewDir(numThreads);
        vector<vector<array<double, TOTAL_NAV_TICKS>>> tmpHealth(numThreads);
        vector<vector<array<double, TOTAL_NAV_TICKS>>> tmpArmor(numThreads);

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
                                trajectoryIndex, curPlayerId, {}, {}
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

            recordSegments(tmpTrajectoryId,
                           tmpSegmentStartTickId,
                           tmpSegmentCurTickId,
                           tmpSegmentFutureTickId,
                           tmpSegmentTickIds, tmpSegmentPATIds,
                           tmpTickLength, tmpPlayerId,
                           tmpTeamId,
                           tmpPlayerViewDir,
                           tmpHealth,
                           tmpArmor,
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
                               result.trajectoryId.push_back(tmpTrajectoryId[minThreadId][tmpRowId]);
                               result.segmentStartTickId.push_back(tmpSegmentStartTickId[minThreadId][tmpRowId]);
                               result.segmentCurTickId.push_back(tmpSegmentCurTickId[minThreadId][tmpRowId]);
                               result.segmentFutureTickId.push_back(tmpSegmentFutureTickId[minThreadId][tmpRowId]);
                               result.segmentTickIds.push_back(tmpSegmentTickIds[minThreadId][tmpRowId]);
                               result.segmentPATIds.push_back(tmpSegmentPATIds[minThreadId][tmpRowId]);
                               result.tickLength.push_back(tmpTickLength[minThreadId][tmpRowId]);
                               result.playerId.push_back(tmpPlayerId[minThreadId][tmpRowId]);
                               result.playerViewDir.push_back(tmpPlayerViewDir[minThreadId][tmpRowId]);
                               result.health.push_back(tmpHealth[minThreadId][tmpRowId]);
                               result.armor.push_back(tmpArmor[minThreadId][tmpRowId]);
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

        // create a fresh directory to save to
        if (createImages) {
            if (!fs::exists(trainNavDir)) {
                fs::create_directory(trainNavDir);
            }
            for (auto &path: fs::directory_iterator(trainNavDir)) {
                fs::remove(path);
            }
        }


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
        MapState mapState(visPoints);
        mapState = visPoints.getCellVisPoints()[2418].visibleFromCurPoint;
        mapState.saveMapState(outputDir + "/visibleFromMid.png");
        CellBits all1s;
        for (const auto &cellVisPoint: visPoints.getCellVisPoints()) {
            all1s.set(cellVisPoint.cellId, true);
        }
        mapState = all1s;
        mapState.saveMapState(outputDir + "/wholeMap.png");
        CellBits viewAngle = getCellsInFOV(visPoints, visPoints.getCellVisPoints()[16133].topCenter, {0., 0.});
        mapState = viewAngle;
        mapState.saveMapState(outputDir + "/BSideTSpawnToASide.png");

        CellBits leftViewAngle = getCellsInFOV(visPoints, visPoints.getCellVisPoints()[2418].topCenter, {90., 0.});
        mapState = leftViewAngle;
        mapState.saveMapState(outputDir + "/midLeft.png");

        CellBits leftVisibleViewAngle = leftViewAngle;
        leftVisibleViewAngle &= visPoints.getCellVisPoints()[2418].visibleFromCurPoint;
        mapState = leftVisibleViewAngle;
        mapState.saveMapState(outputDir + "/midLeftVisible.png");

        CellBits upViewAngle = getCellsInFOV(visPoints, visPoints.getCellVisPoints()[2418].topCenter, {90., -32.});
        mapState = upViewAngle;
        mapState.saveMapState(outputDir + "/midLeftUp.png");

        Vec3 downPos = visPoints.getCellVisPoints()[16133].topCenter;
        downPos.z += 90;
        CellBits straightDownViewAngle = getCellsInFOV(visPoints, downPos, {90., 90.});
        mapState = straightDownViewAngle;
        mapState.saveMapState(outputDir + "/straightDownTestViewAngle.png");

        CellBits straightUpViewAngle = getCellsInFOV(visPoints, downPos, {90., -90.});
        mapState = straightUpViewAngle;
        mapState.saveMapState(outputDir + "/straightUpTestViewAngle.png");
    }
}