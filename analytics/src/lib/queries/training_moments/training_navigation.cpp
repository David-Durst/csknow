//
// Created by durst on 10/20/22.
//

#include "queries/training_moments/training_navigation.h"
#include "bots/analysis/save_map_state.h"
#include "bots/analysis/vis_geometry.h"

namespace csknow {
    namespace navigation {
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
            vector<vector<int64_t>> tmpNavId;
            vector<vector<int64_t>> tmpSegmentStartTickId;
            vector<vector<int64_t>> tmpSegmentNextTickId;
            vector<vector<int64_t>> tmpSegmentFutureTickId;
            vector<vector<int64_t>> tmpTickLength;
            vector<vector<int64_t>> tmpPlayerId;
            vector<vector<string>> tmpPlayerName;
            vector<vector<array<Vec3, TOTAL_NAV_TICKS>>> tmpPlayerViewDir;
            vector<vector<array<double, TOTAL_NAV_TICKS>>> tmpHealth;
            vector<vector<array<double, TOTAL_NAV_TICKS>>> tmpArmor;
            vector<vector<array<TemporalImageNames, TOTAL_NAV_TICKS>>> tmpImgNames;
            vector<vector<string>> tmpGoalRegionImgName;


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