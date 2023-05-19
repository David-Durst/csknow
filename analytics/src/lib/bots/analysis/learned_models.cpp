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
bool useRealProbT = true, useRealProbCT = true;

void processModelArg(string modelArg) {
    bool runRoundsNoHeuristics = modelArg == "r" || modelArg == "tl";
    bool runRoundsHeuristics = modelArg == "rh" || modelArg == "t";
    bool runRoundsHeuristicsT = modelArg == "rht";
    bool runRoundsHeuristicsCT = modelArg == "rhct";
    if (!runRoundsNoHeuristics && !runRoundsHeuristics && !runRoundsHeuristicsT && !runRoundsHeuristicsCT) {
        std::cout << "invalid test option " << modelArg << std::endl;
        exit(1);
    }
    if (runRoundsNoHeuristics) {
        setAllTeamModelProbabilities(true, ENGINE_TEAM_T);
        setAllTeamModelProbabilities(true, ENGINE_TEAM_CT);
    }
    if (runRoundsHeuristics) {
        setAllTeamModelProbabilities(false, ENGINE_TEAM_T);
        setAllTeamModelProbabilities(false, ENGINE_TEAM_CT);
    }
    if (runRoundsHeuristicsT) {
        setAllTeamModelProbabilities(false, ENGINE_TEAM_T);
        setAllTeamModelProbabilities(true, ENGINE_TEAM_CT);
    }
    if (runRoundsHeuristicsCT) {
        setAllTeamModelProbabilities(true, ENGINE_TEAM_T);
        setAllTeamModelProbabilities(false, ENGINE_TEAM_CT);
    }
}

void setAllTeamModelProbabilities(bool value, TeamId teamId) {
    if (teamId == ENGINE_TEAM_T) {
        useOrderModelProbabilitiesT = false;
        useAggressionModelProbabilitiesT = false;
        useTargetModelProbabilitiesT = false;
        usePlaceAreaModelProbabilitiesT = value;
    }
    else {
        useOrderModelProbabilitiesCT = false;
        useAggressionModelProbabilitiesCT = false;
        useTargetModelProbabilitiesCT = false;
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
