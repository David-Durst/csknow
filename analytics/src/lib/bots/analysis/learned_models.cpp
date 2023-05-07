//
// Created by durst on 4/23/23.
//

#include "bots/analysis/learned_models.h"

bool useOrderModelProbabilitiesT = true;
bool useOrderModelProbabilitiesCT = true;
bool useAggressionModelProbabilitiesT = true;
bool useAggressionModelProbabilitiesCT = true;
bool useTargetModelProbabilitiesT = true;
bool useTargetModelProbabilitiesCT = true;
bool usePlaceAreaModelProbabilitiesT = true;
bool usePlaceAreaModelProbabilitiesCT = true;
bool runAllRounds = false;
bool useRealProbT = false, useRealProbCT = false;

void processModelArg(string modelArg) {
    bool runRoundsNoHeuristics = modelArg == "r";
    bool runRoundsHeuristics = modelArg == "rh";
    bool runRoundsHeuristicsT = modelArg == "rht";
    bool runRoundsHeuristicsCT = modelArg == "rhct";
    if (!runRoundsNoHeuristics && !runRoundsHeuristics && !runRoundsHeuristicsT && !runRoundsHeuristicsCT) {
        std::cout << "invalid test option " << modelArg << std::endl;
        exit(1);
    }
    if (runRoundsHeuristics) {
        setAllTeamModelProbabilities(false, ENGINE_TEAM_T);
        setAllTeamModelProbabilities(false, ENGINE_TEAM_CT);
    }
    if (runRoundsHeuristicsT) {
        setAllTeamModelProbabilities(false, ENGINE_TEAM_T);
    }
    if (runRoundsHeuristicsCT) {
        setAllTeamModelProbabilities(false, ENGINE_TEAM_CT);
    }
}

void setAllTeamModelProbabilities(bool value, TeamId teamId) {
    if (teamId == ENGINE_TEAM_T) {
        useOrderModelProbabilitiesT = value;
        useAggressionModelProbabilitiesT = value;
        useTargetModelProbabilitiesT = value;
        usePlaceAreaModelProbabilitiesT = value;
    }
    else {
        useOrderModelProbabilitiesCT = value;
        useAggressionModelProbabilitiesCT = value;
        useTargetModelProbabilitiesCT = value;
        usePlaceAreaModelProbabilitiesCT = value;
    }
}

bool getOrderModelProbabilities(TeamId teamId) {
    if (teamId == ENGINE_TEAM_T) {
        return useOrderModelProbabilitiesT;
    }
    else {
        return useOrderModelProbabilitiesCT;
    }
}

bool getAggressionModelProbabilities(TeamId teamId) {
    if (teamId == ENGINE_TEAM_T) {
        return useAggressionModelProbabilitiesT;
    }
    else {
        return useAggressionModelProbabilitiesCT;
    }
}

bool getTargetModelProbabilities(TeamId teamId) {
    if (teamId == ENGINE_TEAM_T) {
        return useTargetModelProbabilitiesT;
    }
    else {
        return useTargetModelProbabilitiesCT;
    }
}

bool getPlaceAreaModelProbabilities(TeamId teamId) {
    if (teamId == ENGINE_TEAM_T) {
        return usePlaceAreaModelProbabilitiesT;
    }
    else {
        return usePlaceAreaModelProbabilitiesCT;
    }
}
