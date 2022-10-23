//
// Created by durst on 10/20/22.
//

#include "queries/training_moments/training_navigation.h"
#include "bots/analysis/save_map_state.h"
#include "bots/analysis/vis_geometry.h"
#include "queries/lookback.h"
#include "queries/rolling_window.h"

namespace csknow {
    namespace navigation {
        struct NavTSData {
            int64_t trajectoryId, playerId;
            int64_t segmentStartTickId, segmentCurTickId, segmentFutureTickId;
            vector<int64_t> segmentTickIds, segmentPATIds;
            vector<int64_t> syncIds;
        };

        void recordSegments(vector<vector<int64_t>> & tmpNavId,
                            vector<vector<int64_t>> & tmpSegmentStartTickId,
                            vector<vector<int64_t>> & tmpSegmentCurTickId,
                            vector<vector<int64_t>> & tmpSegmentFutureTickId,
                            vector<vector<vector<int64_t>>> tmpSegmentTickIds,
                            vector<vector<int64_t>> & tmpLength, vector<vector<int64_t>> & tmpPlayerId,
                            vector<vector<string>> & tmpPlayerName, vector<vector<array<Vec3, TOTAL_NAV_TICKS>>> & tmpPlayerViewDir,
                            vector<vector<array<double, TOTAL_NAV_TICKS>>> & tmpHealth, vector<vector<array<double, TOTAL_NAV_TICKS>>> & tmpArmor,
                            vector<vector<array<TemporalImageNames, TOTAL_NAV_TICKS>>> & tmpImgNames,
                            vector<vector<string>> & tmpGoalRegionImgName,
                            int threadNum, const Players & players, const PlayerAtTick & playerAtTick,
                            const vector<NavTSData> & finishedSegmentPerRound) {
            for (const auto & tsData : finishedSegmentPerRound) {
                tmpNavId[threadNum].push_back(tsData.trajectoryId);
                tmpSegmentStartTickId[threadNum].push_back(tsData.segmentStartTickId);
                tmpSegmentCurTickId[threadNum].push_back(tsData.segmentCurTickId);
                tmpSegmentFutureTickId[threadNum].push_back(tsData.segmentFutureTickId);
                tmpLength[threadNum].push_back(tmpSegmentFutureTickId[threadNum].back() - tmpSegmentStartTickId[threadNum].back() + 1);
                tmpPlayerId[threadNum].push_back(tsData.playerId);
                tmpPlayerName[threadNum].push_back(players.name[players.idOffset + tsData.playerId]);
                tmpSegmentTickIds[threadNum].push_back(tsData.segmentTickIds);
                tmpPlayerViewDir[threadNum].push_back({});
                tmpHealth[threadNum].push_back({});
                tmpArmor[threadNum].push_back({});
                tmpImgNames[threadNum].push_back({});
                for (size_t i = 0; i < TOTAL_NAV_TICKS; i++) {
                    tmpHealth[threadNum].back()[i] = playerAtTick.health[tsData.segmentPATIds[i]];
                    tmpArmor[threadNum].back()[i] = playerAtTick.armor[tsData.segmentPATIds[i]];
                }
            }
        }

        TrainingNavigationResult queryTrainingNavigation(const VisPoints & visPoints, const ReachableResult & reachableResult,
                                                         const Players & players,
                                                         const Games & games, const Rounds & rounds,
                                                         const Ticks & ticks, const PlayerAtTick & playerAtTick,
                                                         const NonEngagementTrajectoryResult & nonEngagementTrajectoryResult,
                                                         const string & outputDir) {
            TrainingNavigationResult result;
            string trainNavData = outputDir + "/trainNavData";

            // create a fresh directory to save to
            if (!fs::exists(trainNavData)) {
                fs::create_directory(trainNavData);
            }
            for (auto& path: fs::directory_iterator(trainNavData)) {
                fs::remove(path);
            }

            int numThreads = omp_get_max_threads();
            vector<vector<int64_t>> tmpTickId;
            vector<vector<int64_t>> tmpRoundIds(numThreads);
            vector<vector<int64_t>> tmpRoundStarts(numThreads);
            vector<vector<int64_t>> tmpRoundSizes(numThreads);
            vector<vector<int64_t>> tmpNavId;
            vector<vector<int64_t>> tmpSegmentStartTickId;
            vector<vector<vector<int64_t>>> tmpSegmentPastTickId;
            vector<vector<int64_t>> tmpSegmentCurTickId;
            vector<vector<int64_t>> tmpSegmentFutureTickId;
            vector<vector<int64_t>> tmpTickLength;
            vector<vector<int64_t>> tmpPlayerId;
            vector<vector<string>> tmpPlayerName;
            vector<vector<array<Vec3, TOTAL_NAV_TICKS>>> tmpPlayerViewDir;
            vector<vector<array<double, TOTAL_NAV_TICKS>>> tmpHealth;
            vector<vector<array<double, TOTAL_NAV_TICKS>>> tmpArmor;
            vector<vector<array<TemporalImageNames, TOTAL_NAV_TICKS>>> tmpImgNames;
            vector<vector<string>> tmpGoalRegionImgName;

            // for each round
            // for each tick
            // check when each player is in a region visible to enemy team
            // check when c4 is visible to ct (if not planted)
            // going to sync clock for all players (if you started your trajectory off clock, tough luck, not doing anything until next period)
            // for each sync'd tick:
            //      save all alive player pos
            //      save what each alive player can see
            //      combine all alive players per team, save what each team can see
            //      blur each player if not seen since last sync, otherwise make them a point, then save || combo of enemies
            //      save distance map from each player pos to all other points
            //      for t's, save c4 pos. For ct's, save c4 pos if seen recently. otehrwise keep blurring and save
            //          future: handle bomb planted impact on pos knowledge
            // if a player is in trajectory, start a segment for them if no active segment
            // if a player is in a segment, end it if past segment time
            // clear out at end of round with early termination
//#pragma omp parallel for
            for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
                int threadNum = omp_get_thread_num();
                tmpRoundIds[threadNum].push_back(roundIndex);
                tmpRoundStarts[threadNum].push_back(static_cast<int64_t>(tmpSegmentStartTickId[threadNum].size()));

                TickRates tickRates = computeTickRates(games, rounds, roundIndex);

                map<int64_t, NavTSData> playerToCurTrajectory;
                vector<NavTSData> finishedSegmentPerRound;
                RollingWindow rollingWindow(rounds, ticks, playerAtTick);
                int64_t lastSyncTickId = INVALID_ID;

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
                    bool syncTick = false;
                    if (lastSyncTickId == INVALID_ID ||
                        secondsBetweenTicks(ticks, tickRates, lastSyncTickId, tickIndex) >=
                            PAST_NAV_TICKS_SECONDS_DELTA) {
                        lastSyncTickId = tickIndex;
                        syncTick = true;
                        syncToImageNames.push_back({});
                    }

                    // need to remember one CT and one T players paths so can save team data
                    TemporalImageNames ctImgNames, tImgNames;

                    // pass 1: compute everything that is only per player or per one team
                    for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                         patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                        if (playerAtTick.isAlive[patIndex]) {
                            int64_t playerId = playerAtTick.playerId[patIndex];
                            TeamId teamId = playerAtTick.team[patIndex];

                            // compute player pos
                            const CellVisPoint & playerCellVisPoint = visPoints.getNearestCellVisPoint({
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
                            }
                            else {
                                tPos |= localPos;
                            }

                            // add to team vis
                            if (teamId == ENGINE_TEAM_CT) {
                                ctVis |= playerVis[playerId];
                            }
                            else {
                                tVis |= playerVis[playerId];
                            }

                            MapState mapState(visPoints);
                            if (syncTick) {
                                const TemporalImageNames & imgNames = TemporalImageNames(tickIndex, players.name[playerId],
                                                                                         teamId, outputDir);
                                syncToImageNames.back()[playerId] = imgNames;
                                // save pos
                                mapState.saveNewMapState(localPos, imgNames.playerPos);
                                // save view pos
                                mapState.saveNewMapState(playerVis[playerId], imgNames.playerVis);
                                // save distance matrix
                                mapState.saveNewMapState(reachableResult.scaledCellDistanceMatrix[playerCellVisPoint.cellId],
                                                         imgNames.distanceMap);

                                if (teamId == ENGINE_TEAM_CT) {
                                    ctImgNames = imgNames;
                                }
                                else {
                                    tImgNames = imgNames;
                                }
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
                            const CellVisPoint & playerCellId = visPoints.getCellVisPoints()[playerCellIds[playerId]];
                            // assume seen on first tick
                            if (lastTickPlayerSeenByEnemies.find(playerId) == lastTickPlayerSeenByEnemies.end() ||
                                (teamId == ENGINE_TEAM_CT && tVis[playerCellIds[playerId]]) ||
                                (teamId == ENGINE_TEAM_T && ctVis[playerCellIds[playerId]])) {
                                lastTickPlayerSeenByEnemies[playerId] = tickIndex;
                                MapState posStateForEnemies(visPoints);
                                posStateForEnemies = playerPos[playerId];
                                playerPosForEnemies.insert({playerId, std::move(posStateForEnemies)});
                            }
                            else {
                                playerPosForEnemies.at(playerId).conv(UNIFORM_BLUR_MATRIX);
                            }
                            if (teamId == ENGINE_TEAM_CT) {
                                ctVisToEnemies |= playerPosForEnemies.at(playerId);
                            }
                            else {
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
                    const CellVisPoint & c4CellVisPoint = visPoints.getNearestCellVisPoint({
                        ticks.bombX[tickIndex],
                        ticks.bombY[tickIndex],
                        ticks.bombZ[tickIndex]
                    });
                    CellBits c4Pos;
                    c4Pos.set(c4CellVisPoint.cellId, true);
                    c4PosForT = c4Pos;
                    if (lastTickC4SeenByCT = INVALID_ID || ctVis[c4CellVisPoint.cellId]) {
                        lastTickC4SeenByCT = tickIndex;
                        c4PosForCT = c4Pos;
                    }
                    else {
                        c4PosForCT.conv(UNIFORM_BLUR_MATRIX);
                    }
                    if (syncTick) {
                        c4PosForCT.saveMapState(ctImgNames.c4Pos);
                        c4PosForT.saveMapState(tImgNames.c4Pos);
                    }

                }

            }


            return result;
        }

        void testNavImages(const VisPoints & visPoints, const string & outputDir) {
            MapState mapState(visPoints);
            mapState = visPoints.getCellVisPoints()[2418].visibleFromCurPoint;
            mapState.saveMapState(outputDir + "/visibleFromMid.png");
            CellBits all1s;
            for (const auto & cellVisPoint : visPoints.getCellVisPoints()) {
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
}