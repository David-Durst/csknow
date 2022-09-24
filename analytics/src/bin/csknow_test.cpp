#include <iostream>
#include "load_data.h"
#include "queries/moments/engagement.h"
#include "queries/moments/non_engagement_trajectory.h"
#include "queries/moments/trajectory_segments.h"
namespace fs = std::filesystem;

int main([[maybe_unused]] int argc, [[maybe_unused]] char * argv[]) {
    makeMapBasic();
    std::cout << "Hi" << std::endl;
    Rounds filteredRounds;
    Ticks ticks;
    PlayerAtTick playerAtTick;
    EngagementResult engagementResult;
    NonEngagementTrajectoryResult nonEngagementTrajectoryResult =
            queryNonEngagementTrajectory(filteredRounds, ticks, playerAtTick, engagementResult);
    return 0;
}
