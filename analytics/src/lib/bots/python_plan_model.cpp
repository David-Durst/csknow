//
// Created by durst on 3/1/22.
//
#include "load_save_bot_data.h"
#include "bots/python_plan_model.h"
#include "geometryNavConversions.h"

PythonPlanModel::PythonPlanModel(const nav_mesh::nav_file & navFile) : navFile(navFile) {
    Py_Initialize();

    pName = PyUnicode_DecodeFSDefault(moduleName.c_str());
    /* Error checking of pName left out */

    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, functionName.c_str());
        /* pFunc is a new reference */
        if (!(pFunc && PyCallable_Check(pFunc))) {
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function \"%s\"\n", functionName.c_str());
        }
    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", moduleName.c_str());
    }
}

PythonPlanModel::~PythonPlanModel() {
    Py_XDECREF(pFunc);
    Py_DECREF(pModule);
}

TrainDatasetResult::TimeStepState
PythonPlanModel::serverStateToTimeStepState(int32_t csknowId, ServerState serverState) {
    TrainDatasetResult::TimeStepState timeStepState(navFile.m_area_count);
    const ServerState::Client & curClient = serverState.clients[csknowId];
    timeStepState.team = curClient.team;
    timeStepState.pos = {curClient.lastEyePosX, curClient.lastEyePosY, curClient.lastFootPosZ};
    timeStepState.curArea = navFile.m_area_ids_to_indices.at(
            navFile.get_nearest_area_by_position(vec3Conv(timeStepState.pos)).get_id());

    for (const auto & otherClient : serverState.clients) {
        Vec3 otherPos = {otherClient.lastEyePosX, otherClient.lastEyePosY, otherClient.lastFootPosZ};
        size_t navId = navFile.m_area_ids_to_indices.at(
                navFile.get_nearest_area_by_position(vec3Conv(otherPos)).get_id());
        if (otherClient.team == curClient.team) {
            timeStepState.navStates[navId].numFriends++;
        }
        else {
            timeStepState.navStates[navId].numEnemies++;
        }
    }

    return timeStepState;
}

std::optional<int64_t> PythonPlanModel::GetTargetNavArea(int32_t csknowId, ServerState curState,
                                                         ServerState lastState, ServerState oldState) {

    //vector<string> a = {"id","tick id","round id","source player id","team","cur nav area","cur pos x","cur pos y","cur pos z","cur nav 0 friends","cur nav 0 enemies","cur nav 1 friends","cur nav 1 enemies","cur nav 2 friends","cur nav 2 enemies","cur nav 3 friends","cur nav 3 enemies","cur nav 4 friends","cur nav 4 enemies","cur nav 5 friends","cur nav 5 enemies","cur nav 6 friends","cur nav 6 enemies","cur nav 7 friends","cur nav 7 enemies","cur nav 8 friends","cur nav 8 enemies","cur nav 9 friends","cur nav 9 enemies","cur nav 10 friends","cur nav 10 enemies","last nav area","last pos x","last pos y","last pos z","last nav 0 friends","last nav 0 enemies","last nav 1 friends","last nav 1 enemies","last nav 2 friends","last nav 2 enemies","last nav 3 friends","last nav 3 enemies","last nav 4 friends","last nav 4 enemies","last nav 5 friends","last nav 5 enemies","last nav 6 friends","last nav 6 enemies","last nav 7 friends","last nav 7 enemies","last nav 8 friends","last nav 8 enemies","last nav 9 friends","last nav 9 enemies","last nav 10 friends","last nav 10 enemies","old nav area","old pos x","old pos y","old pos z","old nav 0 friends","old nav 0 enemies","old nav 1 friends","old nav 1 enemies","old nav 2 friends","old nav 2 enemies","old nav 3 friends","old nav 3 enemies","old nav 4 friends","old nav 4 enemies","old nav 5 friends","old nav 5 enemies","old nav 6 friends","old nav 6 enemies","old nav 7 friends","old nav 7 enemies","old nav 8 friends","old nav 8 enemies","old nav 9 friends","old nav 9 enemies","old nav 10 friends","old nav 10 enemies","delta x","delta y","shoot next","crouch next","nav target"};
    //vector<long> b = {1,30,0,1,0,7,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,7,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,7,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,7};
    TrainDatasetResult stateToSerialize;
    stateToSerialize.tickId.push_back(0);
    stateToSerialize.roundId.push_back(0);
    stateToSerialize.sourcePlayerId.push_back(0);
    stateToSerialize.sourcePlayerName.push_back("");
    stateToSerialize.demoName.push_back("");
    stateToSerialize.curState.push_back(serverStateToTimeStepState(csknowId, curState));
    stateToSerialize.lastState.push_back(serverStateToTimeStepState(csknowId, lastState));
    stateToSerialize.oldState.push_back(serverStateToTimeStepState(csknowId, oldState));
    stateToSerialize.plan.push_back({});
    vector<string> argNames = {"id"};
    vector<string> foreignKeyNames = stateToSerialize.getForeignKeyNames();
    vector<string> otherColumnNames = stateToSerialize.getOtherColumnNames();
    argNames.insert(argNames.end(), foreignKeyNames.begin(), foreignKeyNames.end());
    argNames.insert(argNames.end(), otherColumnNames.begin(), otherColumnNames.end());
    vector<TrainDatasetResult::TrainDataContainer> argVals = stateToSerialize.oneLineToVec(0);
    assert(argVals.size() == argNames.size());

    vector<PyObject *> keysToFree, valuesToFree;
    PyObject *pArg, *pValue;

    pArg = PyDict_New();
    for (size_t i = 0; i < argNames.size(); i++) {
        keysToFree.push_back(PyUnicode_FromString(argNames[i].c_str()));
        if (argVals[i].containerType == TrainDatasetResult::ContainerType::intType) {
            valuesToFree.push_back(PyLong_FromLong(argVals[i].intOption));
        }
        else if (argVals[i].containerType == TrainDatasetResult::ContainerType::doubleType) {
            valuesToFree.push_back(PyFloat_FromDouble(argVals[i].doubleOption));
        }
        else {
            valuesToFree.push_back(PyUnicode_FromString(argVals[i].stringOption.c_str()));
        }
        PyDict_SetItem(pArg, keysToFree.back(), valuesToFree.back());
    }
    pValue = PyObject_CallOneArg(pFunc, pArg);
    Py_XDECREF(pFunc);
    pFunc = PyObject_GetAttrString(pModule, functionName.c_str());
    pValue = PyObject_CallOneArg(pFunc, pArg);
    Py_DECREF(pArg);
    for (int i = 0; i < keysToFree.size(); i++) {
        Py_DecRef(keysToFree[i]);
        Py_DecRef(valuesToFree[i]);
    }
    if (pValue != NULL) {
        int64_t result = PyLong_AsLong(pValue);
        Py_DECREF(pValue);
        return result;
    }
    else {
        return {};
    }
}
