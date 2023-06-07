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
        Vec3 getPosRelative(const csknow::feature_store::TeamFeatureStoreResult & traces, size_t relativeTraceIndex) const;
        Vec3 startPos(const csknow::feature_store::TeamFeatureStoreResult & traces) const;
        Vec3 endPos(const csknow::feature_store::TeamFeatureStoreResult & traces) const;
        double distance(const csknow::feature_store::TeamFeatureStoreResult & traces) const;
        vector<Trajectory> cutTrajectory(size_t cutTraceIndex) const;
    };

    struct DTWMatrix {
        vector<double> values;
        size_t n, m;
        DTWMatrix(size_t m, size_t n);
        inline double & get(size_t i, size_t j);
    };

    struct MultiTrajectory {
        vector<Trajectory> trajectories;
        int ctTrajectories, tTrajectories;

        double distance(const csknow::feature_store::TeamFeatureStoreResult & traces) const;
        double fde(const csknow::feature_store::TeamFeatureStoreResult & curTraces, const MultiTrajectory & otherMT,
                   const csknow::feature_store::TeamFeatureStoreResult & otherTraces,
                   map<int, int> agentMapping) const;
        double dtw(const csknow::feature_store::TeamFeatureStoreResult & curTraces, const MultiTrajectory & otherMT,
                   const csknow::feature_store::TeamFeatureStoreResult & otherTraces,
                   map<int, int> agentMapping) const;
        size_t minEndTraceIndex() const;
        virtual double minTime(const csknow::feature_store::TeamFeatureStoreResult & traces) const;
        size_t maxTimeSteps() const;

        virtual ~MultiTrajectory() = default;
    };

    struct CutMultiTrajectories {
        MultiTrajectory preCutMT, postCutMT;
    };
    CutMultiTrajectories cutMultiTrajectory(MultiTrajectory mt);

    struct DenseMultiTrajectory : MultiTrajectory {
        DenseMultiTrajectory(MultiTrajectory mt);
        DenseMultiTrajectory() { }
        double minTime(const csknow::feature_store::TeamFeatureStoreResult & traces) const override;
    };

    vector<DenseMultiTrajectory> densifyMT(MultiTrajectory mt);

    vector<MultiTrajectory> createMTs(const csknow::feature_store::TeamFeatureStoreResult & traces);

    typedef map<int, int> AgentMapping;

    typedef map<int, map<int, vector<AgentMapping>>> CTAliveTAliveToAgentMappingOptions;
    CTAliveTAliveToAgentMappingOptions generateAllPossibleMappings();

    struct MultiTrajectorySimilarityResult {
        MultiTrajectory predictedMT, bestFitGroundTruthMT;
        double dtw, deltaTime, deltaDistance;
        AgentMapping agentMapping;

        MultiTrajectorySimilarityResult(const MultiTrajectory & predictedMT, const vector<MultiTrajectory> & groundTruthMTs,
                                        const csknow::feature_store::TeamFeatureStoreResult & predictedTraces,
                                        const csknow::feature_store::TeamFeatureStoreResult & groundTruthTraces,
                                        CTAliveTAliveToAgentMappingOptions ctAliveTAliveToAgentMappingOptions);
    };

    vector<MultiTrajectorySimilarityResult> computeMultiTrajectorySimilarityForAllPredicted(
            const csknow::feature_store::TeamFeatureStoreResult & predictedTraces,
            const csknow::feature_store::TeamFeatureStoreResult & groundTruthTraces);
}

#endif //CSKNOW_MULTI_TRAJECTORY_SIMILARITY_H
