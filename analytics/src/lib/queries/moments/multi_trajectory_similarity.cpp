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

    vector<Trajectory> Trajectory::cutTrajectory(std::size_t cutTraceIndex) const {
        if (cutTraceIndex <= startTraceIndex || cutTraceIndex >= endTraceIndex) {
            return {*this};
        }
        else {
            Trajectory firstHalf = *this, secondHalf = *this;
            firstHalf.endTraceIndex = cutTraceIndex;
            secondHalf.startTraceIndex = cutTraceIndex + 1;
            return {firstHalf, secondHalf};
        }
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

    size_t MultiTrajectory::minEndTraceIndex() const {
        size_t minIndex = std::numeric_limits<size_t>::max();
        for (const auto & trajectory : trajectories) {
            minIndex = std::min(minIndex, trajectory.endTraceIndex);
        }
        return minIndex;
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

    CutMultiTrajectories cutMultiTrajectory(MultiTrajectory mt) {
        CutMultiTrajectories result;
        result.preCutMT.tTrajectories = 0;
        result.preCutMT.ctTrajectories = 0;
        result.postCutMT.tTrajectories = 0;
        result.postCutMT.ctTrajectories = 0;
        size_t minEndTraceIndex = mt.minEndTraceIndex();
        for (const auto & trajectory : mt.trajectories) {
            if (trajectory.endTraceIndex == minEndTraceIndex) {
                result.preCutMT.trajectories.push_back(trajectory);
                if (trajectory.team == ENGINE_TEAM_T) {
                    result.preCutMT.tTrajectories++;
                }
                else {
                    result.preCutMT.ctTrajectories++;
                }
            }
            else if (trajectory.endTraceIndex > minEndTraceIndex) {
                vector<Trajectory> cutTrajectory = trajectory.cutTrajectory(minEndTraceIndex);
                result.preCutMT.trajectories.push_back(cutTrajectory[0]);
                result.postCutMT.trajectories.push_back(cutTrajectory[1]);
                if (trajectory.team == ENGINE_TEAM_T) {
                    result.preCutMT.tTrajectories++;
                    result.postCutMT.tTrajectories++;
                }
                else {
                    result.preCutMT.ctTrajectories++;
                    result.postCutMT.ctTrajectories++;
                }
            }
            else {
                throw std::runtime_error("invalid min trace index computation");
            }
        }
        return result;
    }

    DenseMultiTrajectory::DenseMultiTrajectory(csknow::multi_trajectory_similarity::MultiTrajectory mt) {
        trajectories = mt.trajectories;
        ctTrajectories = mt.ctTrajectories;
        tTrajectories = mt.tTrajectories;
        size_t minEndTraceIndex = mt.minEndTraceIndex();
        for (const auto & trajectory : mt.trajectories) {
            if (trajectory.endTraceIndex != minEndTraceIndex) {
                throw std::runtime_error("dense multi trajectory not dense as trajectories end at different times");
            }
        }
    }

    double DenseMultiTrajectory::minTime(const csknow::feature_store::TeamFeatureStoreResult &traces) const {
        return gameTicksToSeconds(tickRates,
                                  traces.gameTickNumber[trajectories.front().endTraceIndex] -
                                  traces.gameTickNumber[trajectories.front().startTraceIndex]);
    }

    vector<DenseMultiTrajectory> densifyMT(csknow::multi_trajectory_similarity::MultiTrajectory mt) {
        vector<DenseMultiTrajectory> result;
        MultiTrajectory curMT = mt;
        while (!curMT.trajectories.empty()) {
            CutMultiTrajectories cutMultiTrajectories = cutMultiTrajectory(curMT);
            result.emplace_back(cutMultiTrajectories.preCutMT);
            curMT = cutMultiTrajectories.postCutMT;
        }
        return result;
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
            MultiTrajectory predictedMT, vector<DenseMultiTrajectory> groundTruthDMTs,
            const csknow::feature_store::TeamFeatureStoreResult &predictedTraces,
            const csknow::feature_store::TeamFeatureStoreResult &groundTruthTraces,
            CTAliveTAliveToAgentMappingOptions ctAliveTAliveToAgentMappingOptions) {

        // convert multi-trajecotries into dense multi trajectories
        vector<DenseMultiTrajectory> predictedDMTs = densifyMT(predictedMT);

        sumFDE = 0.;
        sumAbsDeltaTime = 0.;
        sumAbsDeltaDistance = 0.;

        // for each predicted DMT, find best ground truth DMT and add it's result to the sum
        for (const auto & predictedDMT : predictedDMTs) {
            DenseMultiTrajectorySimilarityResult curResult;
            curResult.predictedDMT = predictedDMT;
            curResult.fde = std::numeric_limits<double>::max();
            for (const auto & groundTruthDMT : groundTruthDMTs) {
                if (predictedDMT.tTrajectories != groundTruthDMT.tTrajectories ||
                    predictedDMT.ctTrajectories != groundTruthDMT.ctTrajectories) {
                    continue;
                }
                double minFDEPerDMTPair = std::numeric_limits<double>::max();
                AgentMapping bestMapping;
                for (const auto & agentMapping :
                    ctAliveTAliveToAgentMappingOptions[predictedDMT.ctTrajectories][predictedDMT.tTrajectories]) {
                    double curFDEPerDMTPair = predictedDMT.fde(predictedTraces, groundTruthDMT, groundTruthTraces, agentMapping);
                    if (curFDEPerDMTPair < minFDEPerDMTPair) {
                        minFDEPerDMTPair = curFDEPerDMTPair;
                        bestMapping = agentMapping;
                    }
                }
                if (minFDEPerDMTPair < curResult.fde) {
                    curResult.bestFitGroundTruthDMT = groundTruthDMT;
                    curResult.fde = minFDEPerDMTPair;
                    curResult.agentMapping = bestMapping;
                }
            }
            curResult.deltaTime = curResult.predictedDMT.minTime(predictedTraces) -
                                  curResult.bestFitGroundTruthDMT.minTime(groundTruthTraces);
            curResult.deltaDistance = curResult.predictedDMT.distance(predictedTraces) -
                                      curResult.bestFitGroundTruthDMT.distance(groundTruthTraces);
            sumFDE += curResult.fde;
            sumAbsDeltaTime += std::abs(curResult.deltaTime);
            sumAbsDeltaDistance += std::abs(curResult.deltaDistance);
        }
    }


    vector<MultiTrajectorySimilarityResult> computeMultiTrajectorySimilarityForAllPredicted(
            const csknow::feature_store::TeamFeatureStoreResult & predictedTraces,
            const csknow::feature_store::TeamFeatureStoreResult & groundTruthTraces) {
        vector<MultiTrajectory> predictedMTs = createMTs(predictedTraces),
                                groundTruthMTs = createMTs(groundTruthTraces);

        vector<DenseMultiTrajectory> groundTruthDMTs;
        for (const auto & groundTruthMT : groundTruthMTs) {
            vector<DenseMultiTrajectory> curDMTs = densifyMT(groundTruthMT);
            groundTruthDMTs.insert(groundTruthDMTs.end(), curDMTs.begin(), curDMTs.end());
        }

        CTAliveTAliveToAgentMappingOptions ctAliveTAliveToAgentMappingOptions = generateAllPossibleMappings();

        vector<MultiTrajectorySimilarityResult> result;
        result.reserve(predictedMTs.size());
        for (const auto & predictedMT : predictedMTs) {
            result.emplace_back(predictedMT, groundTruthDMTs, predictedTraces, groundTruthTraces,
                                ctAliveTAliveToAgentMappingOptions);
        }
        return result;
    }
}