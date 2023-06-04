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

        // assume everything is at 128 tick
        TickRates tickRates{128, 128};

        // build start/end distances
        baselineTraceRowToGeneratedTrajectoryData.resize(generatedTraces.size);
        for (size_t baselineTrajectoryIndex = 0; baselineTrajectoryIndex < baselineData.size();
             baselineTrajectoryIndex++) {
            for (size_t baselineTraceIndex = baselineData[baselineTrajectoryIndex].startTraceIndex;
                 baselineTraceIndex <= baselineData[baselineTrajectoryIndex].endTraceIndex; baselineTraceIndex++) {
                for (size_t generatedTrajectoryIndex = 0; generatedTrajectoryIndex < generatedData.size();
                     generatedTrajectoryIndex++) {
                    BaselineTraceRowToGeneratedTrajectoryData pairData{true, 0., 0., {}, {}};
                    size_t generatedStartTraceIndex = generatedData[generatedTrajectoryIndex].startTraceIndex;
                    size_t generatedEndTraceIndex = generatedData[generatedTrajectoryIndex].endTraceIndex;
                    // for each player in baseline, match them to nearest in generated
                    // necessary as players may be in different orders
                    // allow more alive in baseline than generated as model can learn to ignore info but not create it
                    // since players can die at different times in two trajectories but all have to be alive at start
                    // align based on nearest at start and compute end distance based on generated trajectory end
                    // (will be recomputing for all baseline ticks, so don't worry about early termination there)
                    for (int baselineColumnIndex = 0; baselineColumnIndex < csknow::feature_store::maxEnemies; baselineColumnIndex++) {
                        if (!baselineTraces.columnCTData[baselineColumnIndex].alive[baselineTraceIndex]) {
                            continue;
                        }
                        bool foundCTMatch = false, foundTMatch = false;
                        int bestCTGeneratedColumnIndex = INVALID_ID, bestTGeneratedColumnIndex;
                        float ctMinDistance = std::numeric_limits<float>::max(),
                            tMinDistance = std::numeric_limits<float>::max();
                        for (int generatedColumnIndex = 0; generatedColumnIndex < csknow::feature_store::maxEnemies; generatedColumnIndex++) {
                            bool ctGeneratedAlive =
                                    generatedTraces.columnCTData[generatedColumnIndex].alive[generatedEndTraceIndex];
                            bool ctAlreadyMatched =
                                    pairData.ctGeneratedColumnIndexToBaselineColumnIndex.count(generatedColumnIndex) > 0;
                            float newCTDistance = static_cast<float>(computeDistance(
                                    generatedTraces.columnCTData[generatedColumnIndex].footPos[generatedStartTraceIndex],
                                    baselineTraces.columnCTData[baselineColumnIndex].footPos[baselineTraceIndex]));
                            if (ctGeneratedAlive && !ctAlreadyMatched && newCTDistance < ctMinDistance) {
                                ctMinDistance = newCTDistance;
                                bestCTGeneratedColumnIndex = generatedColumnIndex;
                                foundCTMatch = true;
                            }

                            bool tGeneratedAlive =
                                    generatedTraces.columnTData[generatedColumnIndex].alive[generatedStartTraceIndex];
                            bool tAlreadyMatched =
                                    pairData.tGeneratedColumnIndexToBaselineColumnIndex.count(generatedColumnIndex) > 0;
                            float newTDistance = static_cast<float>(computeDistance(
                                    generatedTraces.columnTData[generatedColumnIndex].footPos[generatedStartTraceIndex],
                                    baselineTraces.columnTData[baselineColumnIndex].footPos[baselineTraceIndex]));
                            if (tGeneratedAlive && !tAlreadyMatched && newTDistance < tMinDistance) {
                                tMinDistance = newTDistance;
                                bestTGeneratedColumnIndex = generatedColumnIndex;
                                foundTMatch = true;
                            }
                        }
                        // if not able to match alive players at end, then abort as not valid match at this point
                        if (!foundCTMatch || !foundTMatch) {
                            pairData.valid = false;
                            break;
                        }
                        else {
                            pairData.
                        }
                    }
                    baselineTraceRowToGeneratedTrajectoryData[baselineTraceIndex].push_back(pairData);
                }
            }
        }

        // for each baseline-generated trajectory pair,
        //    1. compute point in baseline that is nearest to generated end, taking earlier point in baseline if tie
        //    2. compute point in baseline that is nearest to generated start, taking later point in baseline if tie
        //         a. note - the start point in baseline must be before the end point
        //    3. compute metrics for each best matching subcomponent of baseline trajectory (do generated trajectory metrics once)
        baselineTrajectoryToGeneratedTrajectoryData.resize(baselineData.size());
        for (size_t baselineTrajectoryIndex = 0; baselineTrajectoryIndex < baselineData.size();
             baselineTrajectoryIndex++) {
            for (size_t generatedTrajectoryIndex = 0; generatedTrajectoryIndex < generatedData.size();
                 generatedTrajectoryIndex++) {
                BaselineTrajectoryToGeneratedTrajectoryData pairData;
                // get closest point in baseline to generated end
                float minEndDistance = std::numeric_limits<float>::max();
                for (size_t baselineTraceIndex = baselineData[baselineTrajectoryIndex].startTraceIndex;
                     baselineTraceIndex <= baselineData[baselineTrajectoryIndex].endTraceIndex; baselineTraceIndex++) {
                    float curEndDistance =
                            baselineTraceRowToGeneratedTrajectoryData[baselineTraceIndex][generatedTrajectoryIndex].endDistance;
                    // < rather than <= so take earliest end
                    if (baselineTraceRowToGeneratedTrajectoryData[baselineTraceIndex][generatedTrajectoryIndex].valid &&
                        curEndDistance < minEndDistance) {
                        minEndDistance = curEndDistance;
                        pairData.bestMatchBaselineEndTraceIndex = baselineTraceIndex;
                    }
                }

                // get closest point in baseline to generated start that is before above nearest to end
                float minStartDistance = std::numeric_limits<float>::max();
                for (size_t baselineTraceIndex = baselineData[baselineTrajectoryIndex].startTraceIndex;
                     baselineTraceIndex <= pairData.bestMatchBaselineEndTraceIndex;
                     baselineTraceIndex++) {
                    float curStartDistance =
                            baselineTraceRowToGeneratedTrajectoryData[baselineTraceIndex][generatedTrajectoryIndex].startDistance;
                    // <= rather than < so take later start
                    if (baselineTraceRowToGeneratedTrajectoryData[baselineTraceIndex][generatedTrajectoryIndex].valid &&
                        curStartDistance <= minStartDistance) {
                        minStartDistance = curStartDistance;
                        pairData.bestMatchBaselineStartTraceIndex = baselineTraceIndex;
                    }
                }

                // compute metrics on baseline trajectory subcomponent best matching generated trajectory

                pairData.baselineTime =
                        gameTicksToSeconds(tickRates,
                                           baselineTraces.gameTickNumber[pairData.bestMatchBaselineEndTraceIndex] -
                                           baselineTraces.gameTickNumber[pairData.bestMatchBaselineStartTraceIndex]);
                pairData.generatedTime =
                        gameTicksToSeconds(tickRates,
                                           generatedTraces.gameTickNumber[generatedData[generatedTrajectoryIndex].endTraceIndex] -
                                           generatedTraces.gameTickNumber[generatedData[generatedTrajectoryIndex].startTraceIndex]);


            }
        }


    }
}