//
// Created by durst on 4/17/23.
//

#ifndef CSKNOW_LEARNED_MODELS_H
#define CSKNOW_LEARNED_MODELS_H

#include "bots/load_save_bot_data.h"

extern bool useOrderModelProbabilitiesT, useOrderModelProbabilitiesCT;
extern bool useAggressionModelProbabilitiesT, useAggressionModelProbabilitiesCT;
extern bool useTargetModelProbabilitiesT, useTargetModelProbabilitiesCT;
extern bool usePlaceAreaModelProbabilitiesT, usePlaceAreaModelProbabilitiesCT;
extern bool runAllRounds;
extern bool useRealProbT, useRealProbCT;
extern bool useUncertainModel;

void processModelArg(string modelArg);
void setAllTeamModelProbabilities(bool value, TeamId teamId);
void setUseUncertainModel(bool value);
bool getOrderModelProbabilities(TeamId teamId);
bool getAggressionModelProbabilities(TeamId teamId);
bool getTargetModelProbabilities(TeamId teamId);
bool getPlaceAreaModelProbabilities(TeamId teamId);
bool getUseUncertainModel();

#endif //CSKNOW_LEARNED_MODELS_H
