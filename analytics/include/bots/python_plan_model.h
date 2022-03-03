//
// Created by durst on 3/1/22.
//
#ifndef CSKNOW_MODEL_PLAN_H
#define CSKNOW_MODEL_PLAN_H
#include "load_save_bot_data.h"
#include "queries/train_dataset.h"
#include <optional>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

class PythonPlanModel {
    PyObject *pName, *pModule, *pFunc;
    string moduleName = "learn_bot.inference";
    string functionName = "infer";
    const nav_mesh::nav_file & navFile;
    TrainDatasetResult::TimeStepState serverStateToTimeStepState(int32_t csknowId, ServerState serverState);

public:
    PythonPlanModel(const nav_mesh::nav_file & navFile);
    ~PythonPlanModel();
    std::optional<int64_t> GetTargetNavArea(int32_t csknowId, ServerState curState,
                                            ServerState lastState, ServerState oldState);
};

#endif //CSKNOW_MODEL_PLAN_H
