//
// Created by durst on 10/19/22.
//

#ifndef CSKNOW_TRAINING_NAVIGATION_H
#define CSKNOW_TRAINING_NAVIGATION_H

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
#include "queries/moments/trajectory_segments.h"
#include "queries/reachable.h"

using std::string;
using std::vector;
using std::set;
using std::unordered_map;
using std::vector;
using std::array;
using std::map;

namespace csknow {
    namespace navigation {
        constexpr int PAST_NAV_TICKS = 4;
        constexpr double PAST_NAV_TICKS_SECONDS_DELTA = 0.250;
        constexpr double PAST_NAV_SECONDS = PAST_NAV_TICKS_SECONDS_DELTA * PAST_NAV_TICKS;
        constexpr int CUR_NAV_TICK = 1;
        constexpr int FUTURE_NAV_TICKS = 1;
        constexpr double FUTURE_NAV_SECONDS = 5.0;
        constexpr int TOTAL_NAV_TICKS = PAST_NAV_TICKS + CUR_NAV_TICK + FUTURE_NAV_TICKS;
        constexpr double POS_DELTA_THRESHOLD = 5.;

        struct TemporalImageNames {
            // known purely from one player's data
            string playerPos;
            string playerVis;
            string playerVisFrom;
            string distanceMap;
            string goalPos;
            // require either all players on one or both teams
            string friendlyPos;
            string friendlyVis;
            string visEnemies;
            string c4Pos;
            TemporalImageNames() {};
            TemporalImageNames(int64_t tickIndex, const string & playerName, TeamId teamId,
                                string outputDir) {
                if (outputDir != "") {
                    outputDir += "/";
                }
                string teamName = teamId == ENGINE_TEAM_T ? TEAM_T_NAME : TEAM_CT_NAME;
                playerPos = outputDir + "playerPos_" + playerName + "_" + std::to_string(tickIndex) + ".png";
                playerVis = outputDir + "playerVis_" + playerName + "_" + std::to_string(tickIndex) + ".png";
                playerVisFrom = outputDir + "playerVisFrom_" + playerName + "_" + std::to_string(tickIndex) + ".png";
                distanceMap = outputDir + "distanceMap_" + playerName + "_" + std::to_string(tickIndex) + ".png";
                goalPos = outputDir + "goalPos_" + playerName + "_" + std::to_string(tickIndex) + ".png";
                friendlyPos = outputDir + "friendlyPos_" + teamName + "_" + std::to_string(tickIndex) + ".png";
                friendlyVis = outputDir + "friendlyVis_" + teamName + "_" + std::to_string(tickIndex) + ".png";
                visEnemies = outputDir + "visEnemies_" + teamName + "_" + std::to_string(tickIndex) + ".png";
                c4Pos = outputDir + "c4Pos_" + teamName + "_" + std::to_string(tickIndex) + ".png";
            }
        };

        enum class MovementBins {
            DECREASE = -1,
            CONSTANT = 0,
            INCREASE = 1,
            NUM_MOVEMENT_BINS
        };
        struct MovementResult {
            MovementBins xResult, yResult;
        };

        class TrainingNavigationResult : public QueryResult {
        public:
            const Players & players;
            string trainNavDir = "/tmp/trainNavData";

            vector<RangeIndexEntry> rowIndicesPerRound;
            vector<int64_t> trajectoryId;
            vector<int64_t> segmentStartTickId;
            vector<int64_t> segmentCurTickId;
            vector<int64_t> segmentFutureTickId;
            vector<array<int64_t, TOTAL_NAV_TICKS>> segmentTickIds;
            vector<array<int64_t, TOTAL_NAV_TICKS>> segmentPATIds;
            vector<int64_t> tickLength;
            vector<int64_t> playerId;
            vector<TeamId> teamId;
            vector<array<Vec2, TOTAL_NAV_TICKS>> playerViewDir;
            vector<array<double, TOTAL_NAV_TICKS>> health;
            vector<array<double, TOTAL_NAV_TICKS>> armor;
            vector<MovementResult> movementResult;

            TrainingNavigationResult(const Players & players) : players(players) {
                variableLength = false;
                nonTemporal = true;
                overlay = true;
            }

            vector<int64_t> filterByForeignKey(int64_t otherTableIndex) override {
                vector<int64_t> result;
                for (int64_t i = rowIndicesPerRound[otherTableIndex].minId; i <= rowIndicesPerRound[otherTableIndex].maxId; i++) {
                    if (i == -1) {
                        continue;
                    }
                    result.push_back(i);
                }
                return result;
            }

            void oneLineToCSV(int64_t index, stringstream & ss) override {
                ss << index << "," << segmentCurTickId[index] << "," << trajectoryId[index] << ","
                   << playerId[index] << "," << players.name[players.idOffset + playerId[index]];

                for (size_t i = 0; i < TOTAL_NAV_TICKS; i++) {
                    // empty output dir as may load result from different directory
                    TemporalImageNames imgNames(segmentTickIds[index][i], players.name[playerId[index]],
                                               teamId[index], "");
                    ss << "," << playerViewDir[index][i].x << "," << playerViewDir[index][i].y
                       << "," << health[index][i] << "," << armor[index][i]
                       << "," << imgNames.playerPos
                       << "," << imgNames.playerVis
                       << "," << imgNames.playerVisFrom
                       << "," << imgNames.distanceMap
                       << "," << imgNames.goalPos
                       << "," << imgNames.friendlyPos
                       << "," << imgNames.friendlyVis
                       << "," << imgNames.visEnemies
                       << "," << imgNames.c4Pos;
                }
                ss << "," << enumAsInt(movementResult[index].xResult)
                   << "," << enumAsInt(movementResult[index].yResult);

                ss << std::endl;
            }

            vector<string> getForeignKeyNames() override {
                return {"tick id", "trajectory id", "player id"};
            }

            vector<string> getOtherColumnNames() override {
                vector<string> result;
                result.push_back("player name");
                for (int i = -1*PAST_NAV_TICKS; i <= FUTURE_NAV_TICKS; i++) {
                    result.push_back("player view dir x (t" + toSignedIntString(i, true) + ")");
                    result.push_back("player view dir y (t" + toSignedIntString(i, true) + ")");
                    result.push_back("health (t" + toSignedIntString(i, true) + ")");
                    result.push_back("armor (t" + toSignedIntString(i, true) + ")");
                    result.push_back("player pos (t" + toSignedIntString(i, true) + ")");
                    result.push_back("player vis (t" + toSignedIntString(i, true) + ")");
                    result.push_back("player vis from (t" + toSignedIntString(i, true) + ")");
                    result.push_back("distance map (t" + toSignedIntString(i, true) + ")");
                    result.push_back("goal pos (t" + toSignedIntString(i, true) + ")");
                    result.push_back("friendly pos (t" + toSignedIntString(i, true) + ")");
                    result.push_back("friendly vis (t" + toSignedIntString(i, true) + ")");
                    result.push_back("vis enemies (t" + toSignedIntString(i, true) + ")");
                    result.push_back("c4 pos (t" + toSignedIntString(i, true) + ")");
                }
                result.push_back("movement result x");
                result.push_back("movement result y");
                return result;
            }

        };

        TrainingNavigationResult queryTrainingNavigation(const VisPoints & visPoints, const ReachableResult & reachableResult,
                                                         const Players & players,
                                                         const Games & games, const Rounds & rounds,
                                                         const Ticks & ticks, const PlayerAtTick & playerAtTick,
                                                         const NonEngagementTrajectoryResult & nonEngagementTrajectoryResult,
                                                         const string & outputDir, bool createImages);
        void testNavImages(const VisPoints & visPoints, const string & outputDir);
    }
}

#endif //CSKNOW_TRAINING_NAVIGATION_H
