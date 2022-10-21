//
// Created by durst on 10/20/22.
//

#include "queries/training_moments/training_navigation.h"
#include "bots/analysis/save_map_state.h"

namespace csknow {
    namespace navigation {
        TrainingNavigationResult queryTrainingNavigation(const VisPoints & visPoints /*, const Players & players,
                                                         const Games & games, const Rounds & rounds,
                                                         const Ticks & ticks, const PlayerAtTick & playerAtTick,
                                                         const NonEngagementTrajectoryResult & nonEngagementTrajectoryResult*/) {
            TrainingNavigationResult result;

            MapState mapState(visPoints);
            mapState = visPoints.getCellVisPoints()[16695].visibleFromCurPoint;
            mapState.saveMapState("/home/durst/dev/csknow/analytics/test.png");
            CellBits all1s;
            for (const auto & cellVisPoint : visPoints.getCellVisPoints()) {
                all1s.set(cellVisPoint.cellId, true);
            }
            mapState = all1s;
            mapState.saveMapState("/home/durst/dev/csknow/analytics/test2.png");


            return result;
        }
    }
}