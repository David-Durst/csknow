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

        TrainingNavigationResult queryTrainingNavigation(const VisPoints & visPoints, const Players & players,
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
            //      for t's, save c4 pos. For ct's, save c4 pos if seen recently or planted. otehrwise keep blurring and save
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

                vector<map<int64_t, string>> syncToPosImageNames;
                vector<map<int64_t, string>> syncToVisImageNames;
                vector<map<TeamId, string>> syncToTeamVisImageNames;
                vector<map<int64_t, string>> syncToPosForEnemiesImageNames;
                vector<map<int64_t, string>> syncToDistanceMapImageNames;
                vector<map<TeamId, string>> syncToTeamC4ImageNames;

                map<int64_t, int64_t> lastTickPlayerSeenByEnemies;
                map<int64_t, int64_t>

                for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
                     tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {
                    map<int64_t, int64_t> curPlayerToPAT = rollingWindow.getPATIdForPlayerId(tickIndex);

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