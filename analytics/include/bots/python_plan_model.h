//
// Created by durst on 3/1/22.
//
#ifndef CSKNOW_MODEL_PLAN_H
#define CSKNOW_MODEL_PLAN_H
#define PY_SSIZE_T_CLEAN
#include <Python.h>

class PythonPlanModel {
    PyObject *pName, *pModule, *pFunc;
    PyObject *pArg, *pValue;
    string moduleName = "learn_bot.inference";
    string functionName = "infer";

public:
    PythonPlanModel();
    ~PythonPlanModel();
};

#endif //CSKNOW_MODEL_PLAN_H
