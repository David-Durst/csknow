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
    };

    struct DTWMatrix {
        vector<double> values;
        size_t n, m;
        DTWMatrix(size_t m, size_t n);
        inline double & get(size_t i, size_t j);
        inline double get(size_t i, size_t j) const;
        void print() const;
    };

    struct DTWStepOptionComponent {
        size_t iOffset, jOffset, weight;
    };

    struct DTWStepOptions {
        // oldest component comes first
        vector<vector<DTWStepOptionComponent>> options;
    };

    struct DTWResult {
        vector<pair<size_t, size_t>> matchedIndices;
        double cost = 0.;
    };

    const vector<double> percentiles{0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.};
    struct MultiTrajectory {
        vector<Trajectory> trajectories;
        int ctTrajectories, tTrajectories;
        int64_t roundId;

        double distance(const csknow::feature_store::TeamFeatureStoreResult & traces) const;
        double fde(const csknow::feature_store::TeamFeatureStoreResult & curTraces, const MultiTrajectory & otherMT,
                   const csknow::feature_store::TeamFeatureStoreResult & otherTraces,
                   map<int, int> agentMapping) const;
        DTWResult dtw(const csknow::feature_store::TeamFeatureStoreResult & curTraces, const MultiTrajectory & otherMT,
                      const csknow::feature_store::TeamFeatureStoreResult & otherTraces,
                      map<int, int> agentMapping, DTWStepOptions stepOptions) const;
        double percentileADE(const csknow::feature_store::TeamFeatureStoreResult & curTraces,
                             const MultiTrajectory & otherMT,
                             const csknow::feature_store::TeamFeatureStoreResult & otherTraces,
                             map<int, int> agentMapping) const;
        virtual double minTime(const csknow::feature_store::TeamFeatureStoreResult & traces) const;
        virtual double maxTime(const csknow::feature_store::TeamFeatureStoreResult & traces) const;
        size_t maxTimeSteps() const;
        size_t startTraceIndex() const;
        size_t maxEndTraceIndex() const;

        virtual ~MultiTrajectory() = default;
    };

    vector<MultiTrajectory> createMTs(const csknow::feature_store::TeamFeatureStoreResult & traces);

    typedef map<int, int> AgentMapping;

    typedef map<int, map<int, vector<AgentMapping>>> CTAliveTAliveToAgentMappingOptions;
    CTAliveTAliveToAgentMappingOptions generateAllPossibleMappings();

    struct MultiTrajectorySimilarityMetricData {
        MultiTrajectory mt;
        string name;
        DTWResult dtwResult;
        double deltaTime, deltaDistance;
        AgentMapping agentMapping;

    };

    enum class MetricType {
        UnconstrainedDTW = 0,
        SlopeConstrainedDTW,
        ADE
    };

    struct MultiTrajectorySimilarityResult {
        MultiTrajectory predictedMT;
        string predictedMTName;
        MultiTrajectorySimilarityMetricData unconstrainedDTWData, slopeConstrainedDTWData, adeData;

        const MultiTrajectorySimilarityMetricData & getDataByType(MetricType metricType) const;

        MultiTrajectorySimilarityResult(const MultiTrajectory & predictedMT, const vector<MultiTrajectory> & groundTruthMTs,
                                        const csknow::feature_store::TeamFeatureStoreResult & predictedTraces,
                                        const csknow::feature_store::TeamFeatureStoreResult & groundTruthTraces,
                                        CTAliveTAliveToAgentMappingOptions ctAliveTAliveToAgentMappingOptions);
    };

    struct TraceSimilarityResult {
        vector<MultiTrajectorySimilarityResult> result;
        TraceSimilarityResult(const csknow::feature_store::TeamFeatureStoreResult & predictedTraces,
                              const csknow::feature_store::TeamFeatureStoreResult & groundTruthTraces);
        void toHDF5(const string &filePath);
    };
}

#endif //CSKNOW_MULTI_TRAJECTORY_SIMILARITY_H
