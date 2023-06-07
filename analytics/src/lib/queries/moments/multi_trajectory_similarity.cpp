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

    DTWMatrix::DTWMatrix(std::size_t n, std::size_t m) : values(n * m, std::numeric_limits<double>::infinity()),
        n(n), m(m) { }

    double &DTWMatrix::get(std::size_t i, std::size_t j) { return values[i*m + j]; }

    DTWResult MultiTrajectory::dtw(const csknow::feature_store::TeamFeatureStoreResult & curTraces,
                                   const csknow::multi_trajectory_similarity::MultiTrajectory & otherMT,
                                   const csknow::feature_store::TeamFeatureStoreResult & otherTraces,
                                   map<int, int> agentMapping) const {
        DTWResult result;
        size_t curLength = maxTimeSteps();
        size_t otherLength = otherMT.maxTimeSteps();

        DTWMatrix dtwMatrix(curLength+1, otherLength+1);
        dtwMatrix.get(0, 0) = 0.;

        for (size_t i = 1; i <= curLength; i++) {
            for (size_t j = 1; j <= otherLength; j++) {
                double cost = 0;
                for (const auto & [curAgentIndex, otherAgentIndex] : agentMapping) {
                    // indexing for matrix is offset by 1 to indexing for data as need 0,0 to be before first step
                    cost += computeDistance(trajectories[curAgentIndex].getPosRelative(curTraces, i-1),
                                            otherMT.trajectories[otherAgentIndex].getPosRelative(otherTraces, j-1));
                }
                dtwMatrix.get(i, j) = cost + std::min(dtwMatrix.get(i-1, j),
                                                      std::min(dtwMatrix.get(i, j-1), dtwMatrix.get(i-1, j-1)));
            }
        }
        result.cost = dtwMatrix.get(curLength - 1, otherLength - 1);

        size_t i = curLength - 1, j = otherLength - 1;
        while (i > 0 && j > 0) {
            result.matchedIndices.push_back({i, j});
            double priorI = dtwMatrix.get(i-1, j);
            double priorJ = dtwMatrix.get(i, j-1);
            double priorIJ = dtwMatrix.get(i-1, j-1);
            if (priorI < priorJ && priorI < priorIJ) {
                i--;
            }
            else if (priorJ < priorIJ) {
                j--;
            }
            else {
                i--;
                j--;
            }
        }
        std::reverse(result.matchedIndices.begin(), result.matchedIndices.end());

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

    MultiTrajectorySimilarityResult::MultiTrajectorySimilarityResult(
            const MultiTrajectory & predictedMT, const vector<MultiTrajectory> & groundTruthMTs,
            const csknow::feature_store::TeamFeatureStoreResult &predictedTraces,
            const csknow::feature_store::TeamFeatureStoreResult &groundTruthTraces,
            CTAliveTAliveToAgentMappingOptions ctAliveTAliveToAgentMappingOptions) {

        // for each predicted MT, find best ground truth MT and add it's result to the sum
        this->predictedMT = predictedMT;
        predictedMTName = predictedTraces.testName[predictedMT.startTraceIndex()] + "_rId_" +
                std::to_string(predictedTraces.roundId[predictedMT.startTraceIndex()]) + "_rNum_" +
                std::to_string(predictedTraces.roundNumber[predictedMT.startTraceIndex()]);
        dtwResult.cost = std::numeric_limits<double>::max();
        for (const auto & groundTruthMT : groundTruthMTs) {
            if (predictedMT.tTrajectories != groundTruthMT.tTrajectories ||
                predictedMT.ctTrajectories != groundTruthMT.ctTrajectories) {
                continue;
            }
            for (const auto & agentMapping :
                ctAliveTAliveToAgentMappingOptions[predictedMT.ctTrajectories][predictedMT.tTrajectories]) {
                DTWResult curDTWResult = predictedMT.dtw(predictedTraces, groundTruthMT, groundTruthTraces, agentMapping);
                if (curDTWResult.cost < dtwResult.cost) {
                    dtwResult = curDTWResult;
                    bestAgentMapping = agentMapping;
                    bestFitGroundTruthMT = groundTruthMT;
                }
            }
        }
        bestFitGroundTruthMTName = groundTruthTraces.testName[bestFitGroundTruthMT.startTraceIndex()] + "_rId_" +
                std::to_string(groundTruthTraces.roundId[bestFitGroundTruthMT.startTraceIndex()]) + "_rNum_" +
                std::to_string(groundTruthTraces.roundNumber[bestFitGroundTruthMT.startTraceIndex()]);
        deltaTime = predictedMT.minTime(predictedTraces) - bestFitGroundTruthMT.minTime(groundTruthTraces);
        deltaDistance = predictedMT.distance(predictedTraces) - bestFitGroundTruthMT.distance(groundTruthTraces);
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

    void TraceSimilarityResult::toHDF5(const std::string &filePath) {
        vector<string> predictedNames, bestFitGroundTruthNames;
        vector<int64_t> predictedRoundIds, bestFitGroundTruthRoundIds;
        vector<size_t> predictedStartTraceIndex, predictedEndTraceIndex,
            bestFitGroundTruthStartTraceIndex, bestFitGroundTruthEndTraceIndex;
        vector<double> dtwCost, deltaTime, deltaDistance;
        vector<size_t> startDTWMatchedIndices, lengthDTWMatchedIndices, firstMatchedIndex, secondMatchedIndex;

        for (const auto & mtSimilarityResult : result) {
            predictedNames.push_back(mtSimilarityResult.predictedMTName);
            bestFitGroundTruthNames.push_back(mtSimilarityResult.bestFitGroundTruthMTName);
            predictedRoundIds.push_back(mtSimilarityResult.predictedMT.roundId);
            bestFitGroundTruthRoundIds.push_back(mtSimilarityResult.bestFitGroundTruthMT.roundId);
            predictedStartTraceIndex.push_back(mtSimilarityResult.predictedMT.startTraceIndex());
            predictedEndTraceIndex.push_back(mtSimilarityResult.predictedMT.maxEndTraceIndex());
            bestFitGroundTruthStartTraceIndex.push_back(mtSimilarityResult.bestFitGroundTruthMT.startTraceIndex());
            bestFitGroundTruthEndTraceIndex.push_back(mtSimilarityResult.bestFitGroundTruthMT.maxEndTraceIndex());
            dtwCost.push_back(mtSimilarityResult.dtwResult.cost);
            deltaTime.push_back(mtSimilarityResult.deltaTime);
            deltaDistance.push_back(mtSimilarityResult.deltaDistance);
            startDTWMatchedIndices.push_back(firstMatchedIndex.size());
            for (const auto & matchedIndices : mtSimilarityResult.dtwResult.matchedIndices) {
                firstMatchedIndex.push_back(matchedIndices.first);
                secondMatchedIndex.push_back(matchedIndices.second);
            }
            lengthDTWMatchedIndices.push_back(firstMatchedIndex.size() - startDTWMatchedIndices.back());
        }

        HighFive::File file(filePath, HighFive::File::Overwrite);
        file.createDataSet("/data/predicted name", predictedNames);
        file.createDataSet("/data/best fit ground truth name", bestFitGroundTruthNames);
        file.createDataSet("/data/predicted round id", predictedRoundIds);
        file.createDataSet("/data/best fit ground truth round id", bestFitGroundTruthRoundIds);
        file.createDataSet("/data/predicted start trace index", predictedStartTraceIndex);
        file.createDataSet("/data/predicted end trace index", predictedEndTraceIndex);
        file.createDataSet("/data/best fit ground truth start trace index", bestFitGroundTruthStartTraceIndex);
        file.createDataSet("/data/best fit ground truth end trace index", bestFitGroundTruthEndTraceIndex);
        file.createDataSet("/data/dtw cost", dtwCost);
        file.createDataSet("/data/delta time", deltaTime);
        file.createDataSet("/data/delta distance", deltaTime);
        file.createDataSet("/data/start dtw matched indices", startDTWMatchedIndices);
        file.createDataSet("/data/length dtw matched indices", startDTWMatchedIndices);
        file.createDataSet("/extra/first matched index", firstMatchedIndex);
        file.createDataSet("/extra/second matched index", secondMatchedIndex);

    }
}