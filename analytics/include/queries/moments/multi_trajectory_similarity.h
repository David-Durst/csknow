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
#include <filesystem>
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
    constexpr size_t num_similar_trajectory_matches = 6;
    struct MultiTrajectory {
        // the trace batch containing the data for this multi-trajectory
        std::optional<std::reference_wrapper<const csknow::feature_store::TeamFeatureStoreResult>> traceBatch;
        vector<Trajectory> trajectories;
        int ctTrajectories, tTrajectories;
        int64_t roundId;

        double distance() const;
        double fde(const MultiTrajectory & otherMT, map<int, int> agentMapping) const;
        DTWResult dtw(const MultiTrajectory & otherMT, map<int, int> agentMapping, DTWStepOptions stepOptions) const;
        DTWResult percentileADE(const MultiTrajectory & otherMT, map<int, int> agentMapping) const;
        double minTime() const;
        double maxTime() const;
        size_t maxTimeSteps() const;
        size_t startTraceIndex() const;
        size_t maxEndTraceIndex() const;

    };

    void createMTs(const csknow::feature_store::TeamFeatureStoreResult & traces, vector<MultiTrajectory> & mts);

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
        PercentileADE
    };

    struct MultiTrajectorySimilarityResult {
        MultiTrajectory predictedMT;
        string predictedMTName;
        vector<MultiTrajectorySimilarityMetricData> unconstrainedDTWDataMatches, slopeConstrainedDTWDataMatches,
            adeDataMatches;

        const vector<MultiTrajectorySimilarityMetricData> & getDataByType(MetricType metricType) const;

        void filterTopDataMatches();

        MultiTrajectorySimilarityResult(const MultiTrajectory & predictedMT, const vector<MultiTrajectory> & groundTruthMTs,
                                        CTAliveTAliveToAgentMappingOptions ctAliveTAliveToAgentMappingOptions,
                                        std::optional<std::reference_wrapper<const set<int64_t>>> validGroundTruthRoundIds);
        MultiTrajectorySimilarityResult() { }
    };

    struct TraceSimilarityResult {
        vector<MultiTrajectorySimilarityResult> result;
        TraceSimilarityResult(const vector<csknow::feature_store::TeamFeatureStoreResult> & predictedTraces,
                              const vector<csknow::feature_store::TeamFeatureStoreResult> & groundTruthTraces,
                              std::optional<std::reference_wrapper<const set<int64_t>>> validPredictedRoundIds,
                              std::optional<std::reference_wrapper<const set<int64_t>>> validGroundTruthRoundIds,
                              const std::filesystem::path & logPath);
        void toHDF5(const string &filePath);
    };
}

#endif //CSKNOW_MULTI_TRAJECTORY_SIMILARITY_H
