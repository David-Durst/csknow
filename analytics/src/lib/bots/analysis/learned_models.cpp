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
