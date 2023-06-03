//
// Created by durst on 6/2/23.
//

#include "queries/moments/trajectory_comparison.h"

namespace csknow::trajectory_comparison {
    void TrajectoryComparison::generateTrajectoryStartEnds(vector<TrajectoryData> & trajectoryData,
                                                           const csknow::feature_store::TeamFeatureStoreResult & traces) {

        int64_t priorRoundId = INVALID_ID;
        for (size_t i = 0; i < traces.roundId.size(); i++) {
            if (priorRoundId != traces.roundId[i]) {
                if (priorRoundId != INVALID_ID) {
                    trajectoryData.back().endTraceIndex = i - 1;
                }
                trajectoryData.push_back({});
                priorRoundId = traces.roundId[i];
                trajectoryData.back().startTraceIndex = i;
            }
        }
        trajectoryData.back().startTraceIndex = traces.roundId.back();
    }

    TrajectoryComparison::TrajectoryComparison(csknow::feature_store::TeamFeatureStoreResult generatedTraces,
                                               csknow::feature_store::TeamFeatureStoreResult baselineTraces) {
        generateTrajectoryStartEnds(generatedData, generatedTraces);
        generateTrajectoryStartEnds(baselineData, baselineTraces);

        // build start/end distances
        baselineRowToGeneratedTrajectoryData.resize(generatedTraces.size);
        for (size_t baselineTrajectoryIndex = 0; baselineTrajectoryIndex < baselineData.size();
             baselineTrajectoryIndex++) {
            for (size_t baselineTraceIndex = baselineData[baselineTrajectoryIndex].startTraceIndex;
                 baselineTraceIndex <= baselineData[baselineTrajectoryIndex].endTraceIndex; baselineTraceIndex++) {
                for (size_t generatedTrajectoryIndex = 0; generatedTrajectoryIndex < generatedData.size();
                     generatedTrajectoryIndex++) {
                    TrajectoryPairData pairData{0., 0.};
                    size_t generatedStartTraceIndex = generatedData[generatedTrajectoryIndex].startTraceIndex;
                    size_t generatedEndTraceIndex = generatedData[generatedTrajectoryIndex].endTraceIndex;
                    for (int columnIndex = 0; columnIndex < csknow::feature_store::maxEnemies; columnIndex++) {
                        pairData.startDistance += static_cast<float>(computeDistance(
                                generatedTraces.columnCTData[columnIndex].footPos[generatedStartTraceIndex],
                                baselineTraces.columnCTData[columnIndex].footPos[baselineTraceIndex]));
                        pairData.endDistance += static_cast<float>(computeDistance(
                                generatedTraces.columnCTData[columnIndex].footPos[generatedEndTraceIndex],
                                baselineTraces.columnCTData[columnIndex].footPos[baselineTraceIndex]));
                    }
                    baselineRowToGeneratedTrajectoryData[baselineTraceIndex].push_back(pairData);
                }
            }
        }

        // for each baseline-generated trajectory pair,
        //    1. compute point in baseline that is nearest to generated end, taking earlier point in baseline if tie
        //    2. compute point in baseline that is nearest to generated start, taking later point in baseline if tie
        //         a. note - the start point in baseline must be before the end point
        //    3. compute metrics for each best matching subcomponent of baseline trajectory (do generated trajectory metrics once)
        for (size_t baselineTrajectoryIndex = 0; baselineTrajectoryIndex < baselineData.size();
             baselineTrajectoryIndex++) {
            for (size_t generatedTrajectoryIndex = 0; generatedTrajectoryIndex < generatedData.size();
                 generatedTrajectoryIndex++) {
                // get closest point in baseline to generated end
                float minEndDistance = std::numeric_limits<float>::max();
                for (size_t baselineTraceIndex = baselineData[baselineTrajectoryIndex].startTraceIndex;
                     baselineTraceIndex <= baselineData[baselineTrajectoryIndex].endTraceIndex; baselineTraceIndex++) {
                    float curEndDistance =
                            baselineRowToGeneratedTrajectoryData[baselineTraceIndex][generatedTrajectoryIndex].endDistance;
                    // < rather than <= so take earliest end
                    if (curEndDistance < minEndDistance) {
                        minEndDistance = curEndDistance;
                        baselineData[baselineTrajectoryIndex].bestBaselineMatchGeneratedEndTraceIndex = baselineTraceIndex;
                    }
                }

                // get closest point in baseline to generated start that is before above nearest to end
                float minStartDistance = std::numeric_limits<float>::max();
                for (size_t baselineTraceIndex = baselineData[baselineTrajectoryIndex].startTraceIndex;
                     baselineTraceIndex <= baselineData[baselineTrajectoryIndex].bestBaselineMatchGeneratedEndTraceIndex;
                     baselineTraceIndex++) {
                    float curStartDistance =
                            baselineRowToGeneratedTrajectoryData[baselineTraceIndex][generatedTrajectoryIndex].startDistance;
                    // <= rather than < so take later start
                    if (curStartDistance <= minStartDistance) {
                        minStartDistance = curStartDistance;
                        baselineData[baselineTrajectoryIndex].bestBaselineMatchGeneratedStartTraceIndex = baselineTraceIndex;
                    }
                }

                // compute metrics on baseline trajectory subcomponent best matching generated trajectory
                //baselineData[baselineTrajectoryIndex].


            }
        }


    }
}