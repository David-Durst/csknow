//
// Created by durst on 3/24/22.
//
#ifndef CSKNOW_NAVMESH_TRAJECTORY_H
#define CSKNOW_NAVMESH_TRAJECTORY_H
#include "load_data.h"
#include "queries/query.h"
#include "indices/spotted.h"
#include "navmesh/nav_file.h"
#include "geometry.h"
#include "enum_helpers.h"
#include <array>
#include <string>
#include <map>
#include <sstream>
using std::string;
using std::map;
using std::array;
#define ENGAGEMENT_SECONDS_RADIUS 1.00
#define WEAPON_RECOIL_SCALE 2.0
#define VIEW_RECOIL_TRACKING 0.45

class NavmeshTrajectoryResult : public QueryResult {
public:
    enum class TrajectoryTarget {
        A_SITE,
        B_SITE
    };

    struct Trajectory {
        TrajectoryTarget target;
        RangeIndexEntry startEndTickIds;
        RangeIndexEntry startEndGameTickNumbers;
        vector<size_t> navMeshArea;
        vector<size_t> navMeshPlace;
    };

    string trajectoryToString(Trajectory trajectory) {
        std::stringstream result;
        result << enumAsInt(trajectory.target)
            << "," << trajectory.startEndTickIds.toCSV()
            << "," << trajectory.startEndGameTickNumbers.toCSV();
        return result.str();
    }

    void trajectoryColumnNames(vector<string> & result) {
        result.push_back("Target");
        result.push_back("Start Tick Id");
        result.push_back("End Tick Id");
        result.push_back("Start Game Tick Number");
        result.push_back("End Game Tick Number");
    }


    string trajectoryToSubTableString(int64_t index, Trajectory trajectory) {
        std::stringstream result;

        for (size_t i = 0; i < trajectory.navMeshArea.size(); i++) {
            result << index
                << "," << i
                << "," << trajectory.navMeshArea[i]
                << "," << trajectory.navMeshPlace[i]
                << "," << navFile.m_places[trajectory.navMeshPlace[i]]
                << std::endl;
        }
        return result.str();
    }

    void trajectorySubTableColumnNames(vector<string> & result) {
        result.push_back("Index");
        result.push_back("Nav Mesh Area Id");
        result.push_back("Nav Mesh Place Id");
        result.push_back("Place Name");
    }


    void timeStepActionOneHotNumCategories(vector<string> & result) {
        //result.push_back(std::to_string(enumAsInt(ActionResult::NUM_ACTION_RESULTS)));
    }

    vector<RangeIndexEntry> trajectoryPerRound;
    vector<int64_t> roundId;
    vector<int64_t> sourcePlayerId;
    vector<string> sourcePlayerName;
    vector<string> demoName;
    vector<Trajectory> trajectory;
    const nav_mesh::nav_file & navFile;

    NavmeshTrajectoryResult(const nav_mesh::nav_file & navFile) : navFile(navFile) {
        this->startTickColumn = -1;
        this->ticksPerEvent = 1;
    }

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) {
        // no filtering on dataset
        vector<int64_t> result;
        return result;
    }

    void oneLineToCSV(int64_t index, stringstream & ss) {
        ss << index
            << "," << roundId[index]
            << "," << sourcePlayerId[index]
            << "," << sourcePlayerName[index]
            << "," << demoName[index]
            << "," << trajectoryToString(trajectory[i])
            << std::endl;
    }

    vector<string> getForeignKeyNames() {
        return {"round id", "source player id"};
    }

    vector<string> getOtherColumnNames() {
        vector<string> result{"source player name", "demo name"};
        trajectoryColumnNames(result);
        return result;
    }
};

NavmeshTrajectoryResult queryNavmeshTrajectoryDataset(const Games & games, const Rounds & rounds,
                                                      const Ticks & ticks, const Players & players,
                                                      const PlayerAtTick & playerAtTick, const nav_mesh::nav_file & navFile);

#endif //CSKNOW_NAVMESH_TRAJECTORY_H