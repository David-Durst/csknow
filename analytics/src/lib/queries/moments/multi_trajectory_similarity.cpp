//
// Created by durst on 6/2/23.
//

#include "queries/moments/multi_trajectory_similarity.h"

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

    double MultiTrajectory::distance(const csknow::feature_store::TeamFeatureStoreResult & traces) const {
        double result = 0.;
        for (const auto & trajectory : trajectories) {
            result += trajectory.distance(traces);
        }
        return result;
    }

    double MultiTrajectory::fde(const csknow::feature_store::TeamFeatureStoreResult & curTraces,
                                const csknow::multi_trajectory_similarity::MultiTrajectory & otherMT,
                                const csknow::feature_store::TeamFeatureStoreResult & otherTraces,
                                map<int, int> agentMapping) const {
        double result = 0.;
        for (const auto & [curAgentIndex, otherAgentIndex] : agentMapping) {
            result += computeDistance(trajectories[curAgentIndex].startPos(curTraces),
                                      otherMT.trajectories[otherAgentIndex].startPos(otherTraces));
            result += computeDistance(trajectories[curAgentIndex].endPos(curTraces),
                                      otherMT.trajectories[otherAgentIndex].endPos(otherTraces));
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

    std::pair<size_t, double>
    DTWStepOptions::getMinComponent(const csknow::multi_trajectory_similarity::DTWMatrix &dtwMatrix, std::size_t i,
                                   std::size_t j) {
        double minStepOptionCost = std::numeric_limits<double>::max();
        size_t minStepOptionIndex = 0;
        for (size_t curStepOptionIndex = 0; curStepOptionIndex < options.size(); curStepOptionIndex++) {
            double curStepOptionCost = 0;
            for (const auto & component : options[curStepOptionIndex]) {
                curStepOptionCost += component.weight * dtwMatrix.get(i-component.iOffset, j-component.jOffset);
            }
            if (curStepOptionCost < minStepOptionCost) {
                minStepOptionCost = curStepOptionCost;
                minStepOptionIndex = curStepOptionIndex;
            }
        }
        return {minStepOptionIndex, minStepOptionCost};
    }

    DTWResult MultiTrajectory::dtw(const csknow::feature_store::TeamFeatureStoreResult & curTraces,
                                   const csknow::multi_trajectory_similarity::MultiTrajectory & otherMT,
                                   const csknow::feature_store::TeamFeatureStoreResult & otherTraces,
                                   map<int, int> agentMapping, DTWStepOptions stepOptions) const {
        DTWResult result;
        size_t curLength = maxTimeSteps();
        size_t otherLength = otherMT.maxTimeSteps();

        // compute distances once, will reuse them for jumps
        DTWMatrix independentDTWMatrix(curLength, otherLength);
        for (size_t i = 1; i <= curLength; i++) {
            for (size_t j = 1; j <= otherLength; j++) {
                for (const auto & [curAgentIndex, otherAgentIndex] : agentMapping) {
                    // indexing for matrix is offset by 1 to indexing for data as need 0,0 to be before first step
                    independentDTWMatrix.get(i, j) = computeDistance(
                            trajectories[curAgentIndex].getPosRelative(curTraces, i - 1),
                            otherMT.trajectories[otherAgentIndex].getPosRelative(otherTraces, j - 1));
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
                if (stepOptions.options[minStepOptionIndex].front().iOffset > i ||
                    stepOptions.options[minStepOptionIndex].front().jOffset > j) {
                    std::cout << "danger" << std::endl;
                }
                i -= stepOptions.options[minStepOptionIndex].front().iOffset;
                j -= stepOptions.options[minStepOptionIndex].front().jOffset;
            }
            std::reverse(result.matchedIndices.begin(), result.matchedIndices.end());
        }
        return result;
    }

    double MultiTrajectory::minTime(const csknow::feature_store::TeamFeatureStoreResult &traces) const {
        double minTime = std::numeric_limits<double>::max();
        for (const auto & trajectory : trajectories) {
            minTime = std::min(minTime, gameTicksToSeconds(tickRates,
                                                           traces.gameTickNumber[trajectory.endTraceIndex] -
                                                           traces.gameTickNumber[trajectory.startTraceIndex]));
        }
        return minTime;
    }

    double MultiTrajectory::maxTime(const csknow::feature_store::TeamFeatureStoreResult &traces) const {
        double maxTime = 0;
        for (const auto & trajectory : trajectories) {
            maxTime = std::max(maxTime, gameTicksToSeconds(tickRates,
                                                           traces.gameTickNumber[trajectory.endTraceIndex] -
                                                           traces.gameTickNumber[trajectory.startTraceIndex]));
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

    vector<MultiTrajectory> createMTs(const csknow::feature_store::TeamFeatureStoreResult & traces) {
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

        // filter to rounds where all player ids are present throughout round
        vector<size_t> validRoundIndexes;
        for (size_t roundIndex = 0; roundIndex < roundEndTraceIndex.size(); roundIndex++) {
            set<size_t> startPlayerIds, endPlayerIds;
            for (size_t playerColumnIndex = 0; playerColumnIndex < csknow::feature_store::maxEnemies; playerColumnIndex++) {
                startPlayerIds.insert(traces.columnCTData[playerColumnIndex].playerId[roundStartTraceIndex[roundIndex]]);
                startPlayerIds.insert(traces.columnTData[playerColumnIndex].playerId[roundStartTraceIndex[roundIndex]]);
                endPlayerIds.insert(traces.columnCTData[playerColumnIndex].playerId[roundEndTraceIndex[roundIndex]]);
                endPlayerIds.insert(traces.columnTData[playerColumnIndex].playerId[roundEndTraceIndex[roundIndex]]);
            }
            if (startPlayerIds != endPlayerIds) {
                std::cout << "skipping round " << roundIndex << " as start/end player ids don't match" << std::endl;
                continue;
            }
            else {
                validRoundIndexes.push_back(roundIndex);
            }
        }

        // for each round create a multi-trajectory with a trajectory for each player from start until not alive
        vector<MultiTrajectory> mts;
        for (const auto & roundIndex : validRoundIndexes) {
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
            MultiTrajectory mt;
            mt.trajectories = trajectories;
            mt.ctTrajectories = ctTrajectories;
            mt.tTrajectories = tTrajectories;
            mt.roundId = traces.roundId[roundStartTraceIndex[roundIndex]];
            mts.push_back(mt);
        }

        return mts;
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

    string getName(const MultiTrajectory & mt, const csknow::feature_store::TeamFeatureStoreResult traces) {
        return traces.testName[mt.startTraceIndex()] + "_rId_" +
            std::to_string(traces.roundId[mt.startTraceIndex()]) +
            "_rNum_" + std::to_string(traces.roundNumber[mt.startTraceIndex()]);
    }

    DTWStepOptions stopOptionsP0Symmetric {{
        // just j
        {{0, 1, 1}, {0, 0, 1}},
        // both i and j
        {{1, 1, 1}, {0, 0, 2}},
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

    MultiTrajectorySimilarityResult::MultiTrajectorySimilarityResult(
            const MultiTrajectory & predictedMT, const vector<MultiTrajectory> & groundTruthMTs,
            const csknow::feature_store::TeamFeatureStoreResult &predictedTraces,
            const csknow::feature_store::TeamFeatureStoreResult &groundTruthTraces,
            CTAliveTAliveToAgentMappingOptions ctAliveTAliveToAgentMappingOptions) {

        // for each predicted MT, find best ground truth MT and add it's result to the sum
        this->predictedMT = predictedMT;
        predictedMTName = getName(predictedMT, predictedTraces);
        unconstrainedDTWData.dtwResult.cost = std::numeric_limits<double>::max();
        slopeConstrainedDTWData.dtwResult.cost = std::numeric_limits<double>::max();
        for (const auto & groundTruthMT : groundTruthMTs) {
            if (predictedMT.tTrajectories != groundTruthMT.tTrajectories ||
                predictedMT.ctTrajectories != groundTruthMT.ctTrajectories) {
                continue;
            }
            for (const auto & agentMapping :
                ctAliveTAliveToAgentMappingOptions[predictedMT.ctTrajectories][predictedMT.tTrajectories]) {
                DTWResult unconstrainedCurDTWResult = predictedMT.dtw(predictedTraces, groundTruthMT, groundTruthTraces,
                                                                      agentMapping, stopOptionsP0Symmetric);
                if (unconstrainedCurDTWResult.cost < unconstrainedDTWData.dtwResult.cost) {
                    unconstrainedDTWData.dtwResult = unconstrainedCurDTWResult;
                    unconstrainedDTWData.agentMapping = agentMapping;
                    unconstrainedDTWData.mt = groundTruthMT;
                }
                DTWResult slopeConstrainedCurDTWResult = predictedMT.dtw(predictedTraces, groundTruthMT, groundTruthTraces,
                                                                         agentMapping, stopOptionsP1_2Symmetric);
                if (slopeConstrainedCurDTWResult.cost < slopeConstrainedDTWData.dtwResult.cost) {
                    slopeConstrainedDTWData.dtwResult = slopeConstrainedCurDTWResult;
                    slopeConstrainedDTWData.agentMapping = agentMapping;
                    slopeConstrainedDTWData.mt = groundTruthMT;
                }
            }
        }
        unconstrainedDTWData.name = getName(unconstrainedDTWData.mt, groundTruthTraces);
        unconstrainedDTWData.deltaTime = predictedMT.maxTime(predictedTraces) - unconstrainedDTWData.mt.maxTime(groundTruthTraces);
        unconstrainedDTWData.deltaDistance = predictedMT.distance(predictedTraces) - unconstrainedDTWData.mt.distance(groundTruthTraces);
        slopeConstrainedDTWData.name = getName(slopeConstrainedDTWData.mt, groundTruthTraces);
        slopeConstrainedDTWData.deltaTime = predictedMT.maxTime(predictedTraces) - slopeConstrainedDTWData.mt.maxTime(groundTruthTraces);
        slopeConstrainedDTWData.deltaDistance = predictedMT.distance(predictedTraces) - slopeConstrainedDTWData.mt.distance(groundTruthTraces);
    }


    TraceSimilarityResult::TraceSimilarityResult(const csknow::feature_store::TeamFeatureStoreResult & predictedTraces,
                                                 const csknow::feature_store::TeamFeatureStoreResult & groundTruthTraces) {
        vector<MultiTrajectory> predictedMTs = createMTs(predictedTraces),
                                groundTruthMTs = createMTs(groundTruthTraces);

        CTAliveTAliveToAgentMappingOptions ctAliveTAliveToAgentMappingOptions = generateAllPossibleMappings();

        result.reserve(predictedMTs.size());
        for (const auto & predictedMT : predictedMTs) {
            result.emplace_back(predictedMT, groundTruthMTs, predictedTraces, groundTruthTraces,
                                ctAliveTAliveToAgentMappingOptions);
        }
    }

    const MultiTrajectorySimilarityMetricData & MultiTrajectorySimilarityResult::getDataByType(
            csknow::multi_trajectory_similarity::MetricType metricType) const {
        switch (metricType) {
            case MetricType::UnconstrainedDTW:
                return unconstrainedDTWData;
            case MetricType::SlopeConstrainedDTW:
                return slopeConstrainedDTWData;
            default:
                return adeData;
        }
    }

    void TraceSimilarityResult::toHDF5(const std::string &filePath) {
        vector<string> predictedNames, bestFitGroundTruthNames;
        vector<int> metricTypes;
        vector<int64_t> predictedRoundIds, bestFitGroundTruthRoundIds;
        vector<size_t> predictedStartTraceIndex, predictedEndTraceIndex,
            bestFitGroundTruthStartTraceIndex, bestFitGroundTruthEndTraceIndex;
        vector<double> dtwCost, deltaTime, deltaDistance;
        vector<string> agentMapping;
        vector<size_t> startDTWMatchedIndices, lengthDTWMatchedIndices, firstMatchedIndex, secondMatchedIndex;

        for (const auto & mtSimilarityResult : result) {
            for (const auto & metricType : {MetricType::UnconstrainedDTW, MetricType::SlopeConstrainedDTW}) {
                const MultiTrajectorySimilarityMetricData & metricData = mtSimilarityResult.getDataByType(metricType);
                predictedNames.push_back(mtSimilarityResult.predictedMTName);
                bestFitGroundTruthNames.push_back(metricData.name);
                metricTypes.push_back(enumAsInt(metricType));
                predictedRoundIds.push_back(mtSimilarityResult.predictedMT.roundId);
                bestFitGroundTruthRoundIds.push_back(metricData.mt.roundId);
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

        HighFive::File file(filePath, HighFive::File::Overwrite);
        file.createDataSet("/data/predicted name", predictedNames);
        file.createDataSet("/data/best fit ground truth name", bestFitGroundTruthNames);
        file.createDataSet("/data/metric type", predictedNames);
        file.createDataSet("/data/predicted round id", predictedRoundIds);
        file.createDataSet("/data/best fit ground truth round id", bestFitGroundTruthRoundIds);
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