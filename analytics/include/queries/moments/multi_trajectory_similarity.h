//
// Created by durst on 6/2/23.
//

#ifndef CSKNOW_MULTI_TRAJECTORY_SIMILARITY_H
#define CSKNOW_MULTI_TRAJECTORY_SIMILARITY_H

#include <set>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <list>
#include <map>
#include "bots/analysis/feature_store_team.h"

namespace csknow::multi_trajectory_similarity {
    struct Trajectory {
        TeamId team;
        size_t startTraceIndex, endTraceIndex;
        int playerColumnIndex;

        Vec3 getPos(const csknow::feature_store::TeamFeatureStoreResult & traces, size_t traceIndex) const;
        Vec3 startPos(const csknow::feature_store::TeamFeatureStoreResult & traces) const;
        Vec3 endPos(const csknow::feature_store::TeamFeatureStoreResult & traces) const;
        double distance(const csknow::feature_store::TeamFeatureStoreResult & traces) const;
        vector<Trajectory> cutTrajectory(size_t cutTraceIndex) const;
    };

    struct MultiTrajectory {
        vector<Trajectory> trajectories;
        int ctTrajectories, tTrajectories;

        double distance(const csknow::feature_store::TeamFeatureStoreResult & traces) const;
        double fde(const csknow::feature_store::TeamFeatureStoreResult & curTraces, const MultiTrajectory & otherMT,
                   const csknow::feature_store::TeamFeatureStoreResult & otherTraces,
                   map<int, int> agentMapping) const;
        size_t minEndTraceIndex() const;
        virtual double minTime(const csknow::feature_store::TeamFeatureStoreResult & traces) const;

        virtual ~MultiTrajectory() = default;
    };

    struct CutMultiTrajectories {
        MultiTrajectory preCutMT, postCutMT;
    };
    CutMultiTrajectories cutMultiTrajectory(MultiTrajectory mt);

    struct DenseMultiTrajectory : MultiTrajectory {
        size_t startTraceIndex, endTraceIndex;

        DenseMultiTrajectory(MultiTrajectory mt);
        DenseMultiTrajectory() { }
        double minTime(const csknow::feature_store::TeamFeatureStoreResult & traces) const override;
    };

    vector<DenseMultiTrajectory> densifyMT(MultiTrajectory mt);

    vector<MultiTrajectory> createMTs(const csknow::feature_store::TeamFeatureStoreResult & traces);

    typedef map<int, int> AgentMapping;

    struct DenseMultiTrajectorySimilarityResult {
        DenseMultiTrajectory predictedDMT, bestFitGroundTruthDMT;
        double fde, deltaTime, deltaDistance;
        AgentMapping agentMapping;
    };


    typedef map<int, map<int, vector<AgentMapping>>> CTAliveTAliveToAgentMappingOptions;
    CTAliveTAliveToAgentMappingOptions generateAllPossibleMappings();

    struct MultiTrajectorySimilarityResult {
        vector<DenseMultiTrajectorySimilarityResult> resultPerDMT;
        double sumFDE, sumAbsDeltaTime, sumAbsDeltaDistance;

        MultiTrajectorySimilarityResult(MultiTrajectory predictedMT, vector<DenseMultiTrajectory> groundTruthDMTs,
                                        const csknow::feature_store::TeamFeatureStoreResult & predictedTraces,
                                        const csknow::feature_store::TeamFeatureStoreResult & groundTruthTraces,
                                        CTAliveTAliveToAgentMappingOptions ctAliveTAliveToAgentMappingOptions);
    };

    vector<MultiTrajectorySimilarityResult> computeMultiTrajectorySimilarityForAllPredicted(
            const csknow::feature_store::TeamFeatureStoreResult & predictedTraces,
            const csknow::feature_store::TeamFeatureStoreResult & groundTruthTraces);
}

#endif //CSKNOW_MULTI_TRAJECTORY_SIMILARITY_H
