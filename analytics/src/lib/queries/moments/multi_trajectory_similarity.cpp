//
// Created by durst on 6/2/23.
//

#include "queries/moments/multi_trajectory_similarity.h"
#include "file_helpers.h"
#include <atomic>
#include <mutex>

namespace csknow::multi_trajectory_similarity {
    // assume everything is at 128 tick
    TickRates tickRates{128, 128};

    Vec3 Trajectory::getPos(const csknow::feature_store::TeamFeatureStoreResult &traces, std::size_t traceIndex) const {
        if (team == ENGINE_TEAM_CT) {
            return traces.columnCTData[playerColumnIndex].footPos[traceIndex];
        }
        else {
            return traces.columnTData[playerColumnIndex].footPos[traceIndex];
        }
    }

    Vec3 Trajectory::getPosRelative(const csknow::feature_store::TeamFeatureStoreResult & traces,
                                    size_t relativeTraceIndex) const {
        size_t traceIndex = std::min(relativeTraceIndex + startTraceIndex, endTraceIndex);
        if (team == ENGINE_TEAM_CT) {
            return traces.columnCTData[playerColumnIndex].footPos[traceIndex];
        }
        else {
            return traces.columnTData[playerColumnIndex].footPos[traceIndex];
        }
    }

    Vec3 Trajectory::startPos(const csknow::feature_store::TeamFeatureStoreResult &traces) const {
        return getPos(traces, startTraceIndex);
    }

    Vec3 Trajectory::endPos(const csknow::feature_store::TeamFeatureStoreResult &traces) const {
        return getPos(traces, endTraceIndex);
    }

    double Trajectory::distance(const csknow::feature_store::TeamFeatureStoreResult &traces) const {
        double result = 0.;
        for (size_t traceIndex = startTraceIndex; traceIndex < endTraceIndex; traceIndex++) {
            result += computeDistance(getPos(traces, traceIndex), getPos(traces, traceIndex + 1));
        }
        return result;
    }

    double MultiTrajectory::distance() const {
        double result = 0.;
        for (const auto & trajectory : trajectories) {
            result += trajectory.distance(traceBatch->get());
        }
        return result;
    }

    double MultiTrajectory::fde(const csknow::multi_trajectory_similarity::MultiTrajectory & otherMT,
                                map<int, int> agentMapping) const {
        double result = 0.;
        for (const auto & [curAgentIndex, otherAgentIndex] : agentMapping) {
            result += computeDistance(trajectories[curAgentIndex].startPos(traceBatch->get()),
                                      otherMT.trajectories[otherAgentIndex].startPos(otherMT.traceBatch->get()));
            result += computeDistance(trajectories[curAgentIndex].endPos(traceBatch->get()),
                                      otherMT.trajectories[otherAgentIndex].endPos(otherMT.traceBatch->get()));
        }
        return result;
    }

    DTWMatrix::DTWMatrix(std::size_t n, std::size_t m) : values((n + 1) * (m + 1), std::numeric_limits<double>::infinity()),
        n(n), m(m) { }

    double &DTWMatrix::get(std::size_t i, std::size_t j) { return values[i*(m + 1) + j]; }

    double DTWMatrix::get(std::size_t i, std::size_t j) const { return values[i*(m + 1) + j]; }

    void DTWMatrix::print() const {
        for (size_t i = 0; i < n+1; i++) {
            for (size_t j = 0; j < m+1; j++) {
                std::cout << get(i, j) << ",";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl << std::endl;
    }

    DTWResult MultiTrajectory::dtw(const csknow::multi_trajectory_similarity::MultiTrajectory & otherMT,
                                   map<int, int> agentMapping, DTWStepOptions stepOptions) const {
        DTWResult result;
        size_t curLength = maxTimeSteps();
        size_t otherLength = otherMT.maxTimeSteps();

        // compute distances once, will reuse them for jumps
        DTWMatrix independentDTWMatrix(curLength, otherLength);
        for (size_t i = 1; i <= curLength; i++) {
            for (size_t j = 1; j <= otherLength; j++) {
                independentDTWMatrix.get(i, j) = 0;
                for (const auto & [curAgentIndex, otherAgentIndex] : agentMapping) {
                    // indexing for matrix is offset by 1 to indexing for data as need 0,0 to be before first step
                    independentDTWMatrix.get(i, j) += computeDistance(
                            trajectories[curAgentIndex].getPosRelative(traceBatch->get(), i - 1),
                            otherMT.trajectories[otherAgentIndex].getPosRelative(otherMT.traceBatch->get(), j - 1));
                }
            }
        }

        DTWMatrix dtwMatrix(curLength, otherLength);
        dtwMatrix.get(0, 0) = 0.;
        for (size_t i = 1; i <= curLength; i++) {
            for (size_t j = 1; j <= otherLength; j++) {
                double minStepOptionCost = std::numeric_limits<double>::infinity();
                for (size_t curStepOptionIndex = 0; curStepOptionIndex < stepOptions.options.size(); curStepOptionIndex++) {
                    double curStepOptionCost = 0;
                    const auto & stepOption = stepOptions.options[curStepOptionIndex];
                    for (size_t componentIndex = 0; componentIndex < stepOption.size(); componentIndex++) {
                        const auto & component = stepOption[componentIndex];
                        if (component.iOffset > i || component.jOffset > j) {
                            curStepOptionCost = std::numeric_limits<double>::infinity();
                        }
                        else if (componentIndex > 0) {
                            curStepOptionCost += component.weight * independentDTWMatrix.get(i-component.iOffset, j-component.jOffset);
                        }
                        else {
                            curStepOptionCost += dtwMatrix.get(i-component.iOffset, j-component.jOffset);
                        }
                    }
                    if (curStepOptionCost < minStepOptionCost) {
                        minStepOptionCost = curStepOptionCost;
                    }
                }
                dtwMatrix.get(i, j) = minStepOptionCost;
            }
        }

        //dtwMatrix.print();

        if (std::isinf(dtwMatrix.get(curLength, otherLength))) {
            result.cost = dtwMatrix.get(curLength, otherLength);
        }
        else {
            result.cost = dtwMatrix.get(curLength, otherLength) / static_cast<double>(curLength + otherLength);

            size_t i = dtwMatrix.n, j = dtwMatrix.m;
            while (i > 0 && j > 0) {
                result.matchedIndices.emplace_back(i, j);
                double minStepOptionCost = std::numeric_limits<double>::infinity();
                size_t minStepOptionIndex = 0;
                for (size_t curStepOptionIndex = 0; curStepOptionIndex < stepOptions.options.size(); curStepOptionIndex++) {
                    const auto & component = stepOptions.options[curStepOptionIndex].front();
                    double curStepOptionCost;
                    if (component.iOffset > i || component.jOffset > j) {
                        curStepOptionCost = std::numeric_limits<double>::infinity();
                    }
                    else {
                        curStepOptionCost = dtwMatrix.get(i-component.iOffset, j-component.jOffset);
                    }
                    if (curStepOptionCost < minStepOptionCost) {
                        minStepOptionCost = curStepOptionCost;
                        minStepOptionIndex = curStepOptionIndex;
                    }
                }
                i -= stepOptions.options[minStepOptionIndex].front().iOffset;
                j -= stepOptions.options[minStepOptionIndex].front().jOffset;
            }
            std::reverse(result.matchedIndices.begin(), result.matchedIndices.end());
        }
        return result;
    }

    DTWResult MultiTrajectory::percentileADE(const csknow::multi_trajectory_similarity::MultiTrajectory &otherMT,
                                             map<int, int> agentMapping) const {
        DTWResult result;
        result.cost = 0;

        size_t curLength = maxTimeSteps();
        size_t otherLength = otherMT.maxTimeSteps();
        vector<size_t> curPercentileIndices, otherPercentileIndices;
        for (const auto & percentile : percentiles) {
            curPercentileIndices.push_back(static_cast<size_t>(curLength * percentile));
            otherPercentileIndices.push_back(static_cast<size_t>(otherLength * percentile));
            for (const auto & [curAgentIndex, otherAgentIndex] : agentMapping) {
                result.cost += computeDistance(
                        trajectories[curAgentIndex].getPosRelative(traceBatch->get(), curPercentileIndices.back()),
                        otherMT.trajectories[otherAgentIndex].getPosRelative(otherMT.traceBatch->get(),
                                                                             otherPercentileIndices.back()));
            }
        }
        result.cost /= static_cast<double>(2 * percentiles.size());

        for (size_t percentileCounter = 1; percentileCounter < percentiles.size(); percentileCounter++) {
            size_t curIndex = curPercentileIndices[percentileCounter - 1];
            size_t otherIndex = otherPercentileIndices[percentileCounter - 1];
            while (curIndex < curPercentileIndices[percentileCounter] ||
                   otherIndex < otherPercentileIndices[percentileCounter]) {
                result.matchedIndices.emplace_back(curIndex, otherIndex);
                if (curIndex < curPercentileIndices[percentileCounter]) {
                    curIndex++;
                }
                if (otherIndex < otherPercentileIndices[percentileCounter]) {
                    otherIndex++;
                }
            }
        }
        // handle 100%, since above loop will end before adding that
        result.matchedIndices.emplace_back(curPercentileIndices.back(), otherPercentileIndices.back());

        return result;
    }

    double MultiTrajectory::minTime() const {
        double minTime = std::numeric_limits<double>::max();
        for (const auto & trajectory : trajectories) {
            minTime = std::min(minTime, gameTicksToSeconds(tickRates,
                                                           traceBatch->get().gameTickNumber[trajectory.endTraceIndex] -
                                                           traceBatch->get().gameTickNumber[trajectory.startTraceIndex]));
        }
        return minTime;
    }

    double MultiTrajectory::maxTime() const {
        double maxTime = 0;
        for (const auto & trajectory : trajectories) {
            maxTime = std::max(maxTime, gameTicksToSeconds(tickRates,
                                                           traceBatch->get().gameTickNumber[trajectory.endTraceIndex] -
                                                           traceBatch->get().gameTickNumber[trajectory.startTraceIndex]));
        }
        return maxTime;
    }

    size_t MultiTrajectory::maxTimeSteps() const {
        size_t maxTimeSteps = 0;
        for (const auto & trajectory : trajectories) {
            maxTimeSteps = std::max(maxTimeSteps, trajectory.endTraceIndex - trajectory.startTraceIndex);
        }
        return maxTimeSteps;
    }

    size_t MultiTrajectory::startTraceIndex() const {
        return trajectories.front().startTraceIndex;
    }

    size_t MultiTrajectory::maxEndTraceIndex() const {
        size_t maxEndTraceIndex = 0;
        for (const auto & trajectory : trajectories) {
            maxEndTraceIndex = std::max(maxEndTraceIndex, trajectory.endTraceIndex);
        }
        return maxEndTraceIndex;
    }

    void createMTs(const csknow::feature_store::TeamFeatureStoreResult & traces, vector<MultiTrajectory> & mts) {
        // get all round starts and ends
        int64_t priorRoundId = INVALID_ID;
        vector<size_t> roundStartTraceIndex, roundEndTraceIndex;
        for (size_t i = 0; i < traces.roundId.size(); i++) {
            if (priorRoundId != traces.roundId[i]) {
                if (priorRoundId != INVALID_ID) {
                    roundEndTraceIndex.push_back(i - 1);
                }
                priorRoundId = traces.roundId[i];
                roundStartTraceIndex.push_back(i);
            }
        }
        roundEndTraceIndex.push_back(traces.roundId.size() - 1);

        // for each round create a multi-trajectory with a trajectory for each player from start until not alive
        for (size_t roundIndex = 0; roundIndex < roundEndTraceIndex.size(); roundIndex++) {
            vector<Trajectory> trajectories;
            int ctTrajectories = 0;
            int tTrajectories = 0;
            for (const auto & columnData : traces.getAllColumnData()) {
                for (int columnDataIndex = 0; columnDataIndex < csknow::feature_store::maxEnemies; columnDataIndex++) {
                    // skip those not alive at start
                    if (!columnData.get()[columnDataIndex].alive[roundStartTraceIndex[roundIndex]]) {
                        continue;
                    }
                    columnData.get()[columnDataIndex].ctTeam[roundStartTraceIndex[roundIndex]] ?
                        ctTrajectories++ : tTrajectories++;
                    Trajectory curTrajectory{
                        columnData.get()[columnDataIndex].ctTeam[roundStartTraceIndex[roundIndex]] ?
                            ENGINE_TEAM_CT : ENGINE_TEAM_T,
                        roundStartTraceIndex[roundIndex], 0, columnDataIndex
                    };
                    bool aliveAtEnd = true;
                    for (size_t curTraceIndex = roundStartTraceIndex[roundIndex];
                         curTraceIndex <= roundEndTraceIndex[roundIndex]; curTraceIndex++) {
                        if (!columnData.get()[columnDataIndex].alive[curTraceIndex]) {
                            curTrajectory.endTraceIndex = curTraceIndex - 1;
                            aliveAtEnd = false;
                            break;
                        }
                    }
                    if (aliveAtEnd) {
                        curTrajectory.endTraceIndex = roundEndTraceIndex[roundIndex];
                    }
                    trajectories.push_back(curTrajectory);
                }
            }
            mts.push_back({traces, trajectories, ctTrajectories, tTrajectories,
                           traces.roundId[roundStartTraceIndex[roundIndex]]});
        }
    }

    struct PartialMapping{
        set<int> usedTargets;
        map<int, int> srcToTgt;
    };

    vector<PartialMapping> computePartialMappingsForCTorT(const vector<PartialMapping> & initMappings,
                                                          int numAlive, int offset) {
        // first do ct mappings
        vector<PartialMapping> partialMappings = initMappings;
        for (int depth = offset; depth < offset + numAlive; depth++) {
            vector<PartialMapping> oldPartialMappings = partialMappings;
            partialMappings.clear();
            for (const auto & partialMapping : oldPartialMappings) {
                for (int curTgt = offset; curTgt < offset + numAlive; curTgt++) {
                    if (partialMapping.usedTargets.count(curTgt) > 0) {
                        continue;
                    }
                    auto newPartialMapping = partialMapping;
                    newPartialMapping.usedTargets.insert(curTgt);
                    newPartialMapping.srcToTgt[depth] = curTgt;
                    partialMappings.push_back(newPartialMapping);
                }
            }
        }
        return partialMappings;
    }

    CTAliveTAliveToAgentMappingOptions generateAllPossibleMappings() {
        CTAliveTAliveToAgentMappingOptions ctAliveTAliveToAgentMapping;
        vector<PartialMapping> initialPartialMappings{{}};
        for (int ctAlive = 0; ctAlive <= csknow::feature_store::maxEnemies; ctAlive++) {
            vector<PartialMapping> ctPartialMappings = computePartialMappingsForCTorT(initialPartialMappings,
                                                                                      ctAlive, 0);
            for (int tAlive = 0; tAlive <= csknow::feature_store::maxEnemies; tAlive++) {
                vector<PartialMapping> partialMappings = computePartialMappingsForCTorT(ctPartialMappings,
                                                                                          tAlive, ctAlive);
                vector<AgentMapping> mappingOptions;
                for (const auto & partialMapping : partialMappings) {
                    mappingOptions.push_back(partialMapping.srcToTgt);
                }
                ctAliveTAliveToAgentMapping[ctAlive][tAlive] = mappingOptions;
            }
        }
        return ctAliveTAliveToAgentMapping;
    }

    string getName(const MultiTrajectory & mt) {
        const csknow::feature_store::TeamFeatureStoreResult & traces = mt.traceBatch->get();
        return traces.testName[mt.startTraceIndex()] + "_rId_" +
            std::to_string(traces.roundId[mt.startTraceIndex()]) +
            "_rNum_" + std::to_string(traces.roundNumber[mt.startTraceIndex()]);
    }

    DTWStepOptions stopOptionsP0Symmetric {{
        // just j
        {{0, 1, 1}, {0, 0, 1}},
        // both i and j
        {{1, 1, 1}, {0, 0, 1}},
        // just i
        {{1, 0, 1}, {0, 0, 1}}
    }};

    DTWStepOptions stopOptionsP1_2Symmetric {{
        // 3x speed j
        {{1, 3, 1}, {0, 2, 2}, {0, 1, 1}, {0, 0, 1}},
        // 2x speed j
        {{1, 2, 1}, {0, 1, 2}, {0, 0, 1}},
        // both i and j
        {{1, 1, 1}, {0, 0, 2}},
        // 2x speed j
        {{2, 1, 1}, {1, 0, 2}, {0, 0, 1}},
        // 3x speed i
        {{3, 1, 1}, {2, 0, 2}, {1, 0, 1}, {0, 0, 1}},
    }};

    constexpr bool useADEForMappingAlignment = true;

    MultiTrajectorySimilarityResult::MultiTrajectorySimilarityResult(
            const MultiTrajectory & predictedMT, const vector<MultiTrajectory> & groundTruthMTs,
            CTAliveTAliveToAgentMappingOptions ctAliveTAliveToAgentMappingOptions,
            std::optional<std::reference_wrapper<const set<int64_t>>> validGroundTruthRoundIds) {

        // for each predicted MT, find best ground truth MT and add it's result to the sum
        this->predictedMT = predictedMT;
        predictedMTName = getName(predictedMT);
        for (const auto & groundTruthMT : groundTruthMTs) {
            if (predictedMT.tTrajectories != groundTruthMT.tTrajectories ||
                predictedMT.ctTrajectories != groundTruthMT.ctTrajectories) {
                continue;
            }
            if (validGroundTruthRoundIds && validGroundTruthRoundIds.value().get().count(groundTruthMT.roundId) == 0) {
                continue;
            }
            MultiTrajectorySimilarityMetricData bestUnconstrainedCurDTWData, bestSlopeConstrainedCurDTWData,
                bestADECurData;
            bestUnconstrainedCurDTWData.dtwResult.cost = std::numeric_limits<double>::infinity();
            bestSlopeConstrainedCurDTWData.dtwResult.cost = std::numeric_limits<double>::infinity();
            bestADECurData.dtwResult.cost = std::numeric_limits<double>::infinity();
            for (const auto & agentMapping :
                ctAliveTAliveToAgentMappingOptions[predictedMT.ctTrajectories][predictedMT.tTrajectories]) {
                if (!useADEForMappingAlignment) {
                    DTWResult unconstrainedCurDTWResult = predictedMT.dtw(groundTruthMT, agentMapping,
                                                                          stopOptionsP0Symmetric);
                    if (unconstrainedCurDTWResult.cost < bestUnconstrainedCurDTWData.dtwResult.cost) {
                        bestUnconstrainedCurDTWData.dtwResult = unconstrainedCurDTWResult;
                        bestUnconstrainedCurDTWData.agentMapping = agentMapping;
                        bestUnconstrainedCurDTWData.mt = groundTruthMT;
                    }
                    DTWResult slopeConstrainedCurDTWResult = predictedMT.dtw(groundTruthMT, agentMapping,
                                                                             stopOptionsP1_2Symmetric);
                    if (slopeConstrainedCurDTWResult.cost < bestSlopeConstrainedCurDTWData.dtwResult.cost) {
                        bestSlopeConstrainedCurDTWData.dtwResult = slopeConstrainedCurDTWResult;
                        bestSlopeConstrainedCurDTWData.agentMapping = agentMapping;
                        bestSlopeConstrainedCurDTWData.mt = groundTruthMT;
                    }

                }
                DTWResult adeCurResult = predictedMT.percentileADE(groundTruthMT, agentMapping);
                if (adeCurResult.cost < bestADECurData.dtwResult.cost) {
                    bestADECurData.dtwResult = adeCurResult;
                    bestADECurData.agentMapping = agentMapping;
                    bestADECurData.mt = groundTruthMT;
                }
            }

            if (useADEForMappingAlignment) {
                DTWResult unconstrainedCurDTWResult = predictedMT.dtw(groundTruthMT, bestADECurData.agentMapping,
                                                                      stopOptionsP0Symmetric);
                bestUnconstrainedCurDTWData.dtwResult = unconstrainedCurDTWResult;
                bestUnconstrainedCurDTWData.agentMapping = bestADECurData.agentMapping;
                bestUnconstrainedCurDTWData.mt = groundTruthMT;
                DTWResult slopeConstrainedCurDTWResult = predictedMT.dtw(groundTruthMT, bestADECurData.agentMapping,
                                                                         stopOptionsP1_2Symmetric);
                bestSlopeConstrainedCurDTWData.dtwResult = slopeConstrainedCurDTWResult;
                bestSlopeConstrainedCurDTWData.agentMapping = bestADECurData.agentMapping;
                bestSlopeConstrainedCurDTWData.mt = groundTruthMT;
            }

            unconstrainedDTWDataMatches.push_back(bestUnconstrainedCurDTWData);
            if (bestSlopeConstrainedCurDTWData.dtwResult.cost < std::numeric_limits<double>::infinity()) {
                slopeConstrainedDTWDataMatches.push_back(bestSlopeConstrainedCurDTWData);
            }
            adeDataMatches.push_back(bestADECurData);
            if (unconstrainedDTWDataMatches.size() > 100) {
                filterTopDataMatches();
            }
        }
        filterTopDataMatches();
        for (auto & unconstrainedDTWData : unconstrainedDTWDataMatches) {
            unconstrainedDTWData.name = getName(unconstrainedDTWData.mt);
            unconstrainedDTWData.deltaTime = predictedMT.maxTime() - unconstrainedDTWData.mt.maxTime();
            unconstrainedDTWData.deltaDistance = predictedMT.distance() - unconstrainedDTWData.mt.distance();
        }
        for (auto & slopeConstrainedDTWData : slopeConstrainedDTWDataMatches) {
            slopeConstrainedDTWData.name = getName(slopeConstrainedDTWData.mt);
            slopeConstrainedDTWData.deltaTime = predictedMT.maxTime() - slopeConstrainedDTWData.mt.maxTime();
            slopeConstrainedDTWData.deltaDistance = predictedMT.distance() - slopeConstrainedDTWData.mt.distance();
        }
        for (auto & adeData : adeDataMatches) {
            adeData.name = getName(adeData.mt);
            adeData.deltaTime = predictedMT.maxTime() - adeData.mt.maxTime();
            adeData.deltaDistance = predictedMT.distance() - adeData.mt.distance();
        }
    }


    TraceSimilarityResult::TraceSimilarityResult(const vector<csknow::feature_store::TeamFeatureStoreResult> & predictedTraces,
                                                 const vector<csknow::feature_store::TeamFeatureStoreResult> & groundTruthTraces,
                                                 std::optional<std::reference_wrapper<const set<int64_t>>> validPredictedRoundIds,
                                                 std::optional<std::reference_wrapper<const set<int64_t>>> validGroundTruthRoundIds,
                                                 const std::filesystem::path & logPath) {
        vector<MultiTrajectory> predictedMTs, groundTruthMTs;
        for (const auto & predictedTraceBatch : predictedTraces) {
            createMTs(predictedTraceBatch, predictedMTs);
        }
        for (const auto & groundTruthTraceBatch : groundTruthTraces) {
            createMTs(groundTruthTraceBatch, groundTruthMTs);
        }

        CTAliveTAliveToAgentMappingOptions ctAliveTAliveToAgentMappingOptions = generateAllPossibleMappings();

        std::mutex similarityMutex;
        vector<bool> valid(omp_get_max_threads(), false);
        vector<int> ctTrajectories(omp_get_max_threads(), 0);
        vector<int> tTrajectories(omp_get_max_threads(), 0);
        vector<int> totalTrajectories(omp_get_max_threads(), 0);
        vector<int> mtIndex(omp_get_max_threads(), -1);
        vector<CSKnowTime> timeAtStart(omp_get_max_threads());
        std::fstream finishedFile (logPath / "finishedSimilarity.log", std::fstream::out);
        finishedFile << "thread,mt index,runtime (s),num alive at start" << std::endl;
        finishedFile.flush();

        std::atomic<size_t> predictedMTsProcessed = 0;
        size_t totalSize = 250;
        result.resize(totalSize/*predictedMTs.size()*/);
#pragma omp parallel for
        for (size_t i = 0; i < totalSize/*predictedMTs.size()*/; i++) {
            const auto & predictedMT = predictedMTs[i];

            similarityMutex.lock();
            valid[omp_get_thread_num()] = true;
            ctTrajectories[omp_get_thread_num()] = predictedMT.ctTrajectories;
            tTrajectories[omp_get_thread_num()] = predictedMT.tTrajectories;
            totalTrajectories[omp_get_thread_num()] = predictedMT.tTrajectories + predictedMT.ctTrajectories;
            mtIndex[omp_get_thread_num()] = i;
            auto curTime = std::chrono::system_clock::now();
            timeAtStart[omp_get_thread_num()] = curTime;
            std::fstream logFile (logPath / "similarityState.log", std::fstream::out);
            for (size_t j = 0; j < valid.size(); j++) {
                if (valid[j]) {
                    std::chrono::duration<double> runtime = curTime - timeAtStart[j];
                    logFile << "thread: " << j << ", mt index: " << mtIndex[j] << ", runtime (s): " << runtime.count() <<
                        ", ct trajectories: " << ctTrajectories[j] << ", t trajectories " << tTrajectories[j] <<
                        ", total trajectories " << totalTrajectories[j] << std::endl;
                }
            }
            logFile.close();
            similarityMutex.unlock();

            if (validPredictedRoundIds && validPredictedRoundIds.value().get().count(predictedMT.roundId) == 0) {
                continue;
            }
            result[i] = MultiTrajectorySimilarityResult(predictedMT, groundTruthMTs, ctAliveTAliveToAgentMappingOptions,
                                                        validGroundTruthRoundIds);
            predictedMTsProcessed++;
            printProgress(predictedMTsProcessed, totalSize/*predictedMTs.size()*/);

            similarityMutex.lock();
            valid[omp_get_thread_num()] = false;
            int threadNum = omp_get_thread_num();
            std::chrono::duration<double> endRuntime = std::chrono::system_clock::now() - timeAtStart[threadNum];
            finishedFile << threadNum << "," << mtIndex[threadNum] << "," << endRuntime.count() << "," << ctTrajectories[threadNum]
                << "," << tTrajectories[threadNum] << "," << totalTrajectories[threadNum] << std::endl;
            finishedFile.flush();
            similarityMutex.unlock();
        }
        finishedFile.close();
    }

    const vector<MultiTrajectorySimilarityMetricData> & MultiTrajectorySimilarityResult::getDataByType(
            csknow::multi_trajectory_similarity::MetricType metricType) const {
        switch (metricType) {
            case MetricType::UnconstrainedDTW:
                return unconstrainedDTWDataMatches;
            case MetricType::SlopeConstrainedDTW:
                return slopeConstrainedDTWDataMatches;
            default:
                return adeDataMatches;
        }
    }

    bool multiTrajectorySimilarityMetricDataComparator(const MultiTrajectorySimilarityMetricData & a,
                                                       const MultiTrajectorySimilarityMetricData & b) {
        return a.dtwResult.cost < b.dtwResult.cost;
    }

    void MultiTrajectorySimilarityResult::filterTopDataMatches() {
        std::sort(unconstrainedDTWDataMatches.begin(), unconstrainedDTWDataMatches.end(),
                  multiTrajectorySimilarityMetricDataComparator);
        unconstrainedDTWDataMatches.resize(std::min(num_similar_trajectory_matches,
                                                    unconstrainedDTWDataMatches.size()));
        std::sort(slopeConstrainedDTWDataMatches.begin(), slopeConstrainedDTWDataMatches.end(),
                  multiTrajectorySimilarityMetricDataComparator);
        slopeConstrainedDTWDataMatches.resize(std::min(num_similar_trajectory_matches,
                                                       slopeConstrainedDTWDataMatches.size()));
        std::sort(adeDataMatches.begin(), adeDataMatches.end(), multiTrajectorySimilarityMetricDataComparator);
        adeDataMatches.resize(std::min(num_similar_trajectory_matches, adeDataMatches.size()));
    }

    string metricTypeToString(MetricType metricType) {
        switch (metricType) {
            case MetricType::UnconstrainedDTW:
                return "Unconstrained DTW";
            case MetricType::SlopeConstrainedDTW:
                return "Slope Constrained DTW";
            default:
                return "Percentile ADE";
        }
    }

    void TraceSimilarityResult::toHDF5(const std::string &filePath) {
        vector<string> predictedNames, bestFitGroundTruthNames;
        vector<string> metricTypes;
        vector<string> predictedTraceBatch, bestFitGroundTruthTraceBatch;
        vector<int64_t> predictedRoundIds, bestFitGroundTruthRoundIds;
        vector<size_t> bestMatchIds, predictedStartTraceIndex, predictedEndTraceIndex,
            bestFitGroundTruthStartTraceIndex, bestFitGroundTruthEndTraceIndex;
        vector<double> dtwCost, deltaTime, deltaDistance;
        vector<string> agentMapping;
        vector<size_t> startDTWMatchedIndices, lengthDTWMatchedIndices, firstMatchedIndex, secondMatchedIndex;

        for (const auto & mtSimilarityResult : result) {
            for (const auto & metricType : {MetricType::UnconstrainedDTW, MetricType::SlopeConstrainedDTW,
                                            MetricType::PercentileADE}) {
                const vector<MultiTrajectorySimilarityMetricData> & metricDataMatches = mtSimilarityResult.getDataByType(metricType);
                for (size_t bestMatchId = 0; bestMatchId < metricDataMatches.size(); bestMatchId++) {
                    const MultiTrajectorySimilarityMetricData & metricData = metricDataMatches[bestMatchId];
                    predictedNames.push_back(mtSimilarityResult.predictedMTName);
                    bestFitGroundTruthNames.push_back(metricData.name);
                    metricTypes.push_back(metricTypeToString(metricType));
                    predictedTraceBatch.push_back(mtSimilarityResult.predictedMT.traceBatch->get().fileName);
                    bestFitGroundTruthTraceBatch.push_back(metricData.mt.traceBatch->get().fileName);
                    predictedRoundIds.push_back(mtSimilarityResult.predictedMT.roundId);
                    bestFitGroundTruthRoundIds.push_back(metricData.mt.roundId);
                    bestMatchIds.push_back(bestMatchId);
                    predictedStartTraceIndex.push_back(mtSimilarityResult.predictedMT.startTraceIndex());
                    predictedEndTraceIndex.push_back(mtSimilarityResult.predictedMT.maxEndTraceIndex());
                    bestFitGroundTruthStartTraceIndex.push_back(metricData.mt.startTraceIndex());
                    bestFitGroundTruthEndTraceIndex.push_back(metricData.mt.maxEndTraceIndex());
                    dtwCost.push_back(metricData.dtwResult.cost);
                    deltaTime.push_back(metricData.deltaTime);
                    deltaDistance.push_back(metricData.deltaDistance);
                    string mappingStr = "";
                    bool first = true;
                    for (const auto & [src, tgt] : metricData.agentMapping) {
                        mappingStr += (!first ? "," : "") + std::to_string(src) + "_" + std::to_string(tgt);
                        first = false;
                    }
                    agentMapping.push_back(mappingStr);
                    startDTWMatchedIndices.push_back(firstMatchedIndex.size());
                    for (const auto & matchedIndices : metricData.dtwResult.matchedIndices) {
                        firstMatchedIndex.push_back(matchedIndices.first);
                        secondMatchedIndex.push_back(matchedIndices.second);
                    }
                    lengthDTWMatchedIndices.push_back(firstMatchedIndex.size() - startDTWMatchedIndices.back());
                }
            }
        }

        HighFive::File file(filePath, HighFive::File::Overwrite);
        file.createDataSet("/data/predicted name", predictedNames);
        file.createDataSet("/data/best fit ground truth name", bestFitGroundTruthNames);
        file.createDataSet("/data/metric type", metricTypes);
        file.createDataSet("/data/predicted trace batch", predictedTraceBatch);
        file.createDataSet("/data/best fit ground truth trace batch", bestFitGroundTruthTraceBatch);
        file.createDataSet("/data/predicted round id", predictedRoundIds);
        file.createDataSet("/data/best fit ground truth round id", bestFitGroundTruthRoundIds);
        file.createDataSet("/data/best match ids", bestMatchIds);
        file.createDataSet("/data/predicted start trace index", predictedStartTraceIndex);
        file.createDataSet("/data/predicted end trace index", predictedEndTraceIndex);
        file.createDataSet("/data/best fit ground truth start trace index", bestFitGroundTruthStartTraceIndex);
        file.createDataSet("/data/best fit ground truth end trace index", bestFitGroundTruthEndTraceIndex);
        file.createDataSet("/data/dtw cost", dtwCost);
        file.createDataSet("/data/delta time", deltaTime);
        file.createDataSet("/data/delta distance", deltaDistance);
        file.createDataSet("/data/agent mapping", agentMapping);
        file.createDataSet("/data/start dtw matched indices", startDTWMatchedIndices);
        file.createDataSet("/data/length dtw matched indices", lengthDTWMatchedIndices);
        file.createDataSet("/extra/first matched index", firstMatchedIndex);
        file.createDataSet("/extra/second matched index", secondMatchedIndex);

    }
}