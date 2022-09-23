//
// Created by durst on 7/6/22.
//

#ifndef CSKNOW_TRAJECTORY_SEGMENTS_H
#define CSKNOW_TRAJECTORY_SEGMENTS_H
#include <string>
#include <set>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <list>
#include <map>
#include "bots/analysis/load_save_vis_points.h"
#include "navmesh/nav_file.h"
#include "load_data.h"
#include "queries/query.h"
#include "geometry.h"
#include "enum_helpers.h"
#include "queries/moments/non_engagement_trajectory.h"
using std::string;
using std::vector;
using std::set;
using std::unordered_map;
using std::vector;
using std::map;
//#define SEGMENT_SECONDS 1.0

class TrajectorySegmentResult : public QueryResult {
public:
    vector<RangeIndexEntry> rowIndicesPerRound;
    vector<int64_t> segmentStartTickId;
    vector<int64_t> segmentEndTickId;
    vector<int64_t> tickLength;
    vector<int64_t> playerId;
    vector<string> playerName;
    vector<Vec2> segmentStart2DPos;
    vector<Vec2> segmentEnd2DPos;

    TrajectorySegmentResult() {
        variableLength = false;
        nonTemporal = true;
        overlay = true;
        overlayLabels = true;
    }

    vector<int64_t> filterByForeignKey(int64_t) override {
        return {};
    }

    void oneLineToCSV(int64_t index, stringstream & ss) override {
        ss << index << "," << segmentStartTickId[index] << "," << segmentEndTickId[index] << ","
           << tickLength[index] << "," << playerId[index] << "," << playerName[index] << ","
           << segmentStart2DPos[index].x << "," << segmentStart2DPos[index].y << ","
           << segmentEnd2DPos[index].x << "," << segmentEnd2DPos[index].y;

        ss << std::endl;
    }

    [[nodiscard]]
    vector<string> getForeignKeyNames() override {
        return {"start tick id", "end tick id", "length", "player id"};
    }

    [[nodiscard]]
    vector<string> getOtherColumnNames() override {
        return {"player name", "segment start 2d pos x", "segment start 2d pos y",
                "segment end 2d pos x", "segment end 2d pos y"};
    }
};


TrajectorySegmentResult queryAllTrajectories(const Players & players, const Games & games, const Rounds & rounds,
                                             const Ticks & ticks, const PlayerAtTick & playerAtTick,
                                             const NonEngagementTrajectoryResult & nonEngagementTrajectoryResult);

#endif //CSKNOW_TRAJECTORY_SEGMENTS_H
