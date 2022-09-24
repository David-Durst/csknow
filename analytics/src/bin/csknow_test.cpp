#include <iostream>
#include <unistd.h>
#include <map>
#include <string>
#include <sstream>
#include <functional>
#include <fstream>
#include <iomanip>
#include <ctime>
#include "load_data.h"
#include "indices/build_indexes.h"
#include "load_cover.h"
#include "load_clusters.h"
#include "queries/velocity.h"
#include "queries/wallers.h"
#include "queries/baiters.h"
#include "queries/netcode.h"
#include "queries/looking.h"
#include "queries/nearest_origin.h"
#include "queries/player_in_cover_edge.h"
#include "queries/team_looking_at_cover_edge_cluster.h"
#include "queries/nonconsecutive.h"
#include "queries/grouping.h"
#include "queries/groupInSequenceOfRegions.h"
#include "queries/base_tables.h"
#include "queries/position_and_wall_view.h"
#include "indices/spotted.h"
#include "queries/nav_visible.h"
#include "queries/nav_danger.h"
#include "queries/distance_to_places.h"
#include "queries/moments/aggression_event.h"
#include "queries/moments/engagement.h"
#include "queries/moments/engagement_per_tick_aim.h"
#include "queries/moments/non_engagement_trajectory.h"
#include "queries/moments/trajectory_segments.h"
#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "httplib.h"
#include <cerrno>
#include "navmesh/nav_file.h"
#include "queries/nav_mesh.h"
#include "queries/reachable.h"
#include <filesystem>
namespace fs = std::filesystem;

using std::map;
using std::string;
using std::reference_wrapper;

void exec(const string & cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::string cmdWithPipe = cmd + " 2>&1";
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmdWithPipe.c_str(), "r"), pclose);
    if (!pipe) {
        std::cerr << "error code: " << strerror(errno) << std::endl;
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    std::cout << result;
}

int main(int argc, char * argv[]) {
    makeMapBasic();
    NonEngagementTrajectoryResult nonEngagementTrajectoryResult =
            queryNonEngagementTrajectory(filteredRounds, ticks, playerAtTick, engagementResult);
    return 0;
}
