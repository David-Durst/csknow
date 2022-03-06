//
// Created by durst on 3/1/22.
//
#include "load_save_bot_data.h"
#include "bots/python_model_interface.h"
#include "geometryNavConversions.h"
#include <filesystem>
#include <file_helpers.h>

TrainDatasetResult::TimeStepState
PythonModelInterface::serverStateToTimeStepState(int32_t csknowId, ServerState serverState) {
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

int64_t PythonModelInterface::GetTargetNavArea(int32_t csknowId, ServerState curState,
                                       ServerState lastState, ServerState oldState) {

    //vector<string> a = {"id","tick id","round id","source player id","team","cur nav area","cur pos x","cur pos y","cur pos z","cur nav 0 friends","cur nav 0 enemies","cur nav 1 friends","cur nav 1 enemies","cur nav 2 friends","cur nav 2 enemies","cur nav 3 friends","cur nav 3 enemies","cur nav 4 friends","cur nav 4 enemies","cur nav 5 friends","cur nav 5 enemies","cur nav 6 friends","cur nav 6 enemies","cur nav 7 friends","cur nav 7 enemies","cur nav 8 friends","cur nav 8 enemies","cur nav 9 friends","cur nav 9 enemies","cur nav 10 friends","cur nav 10 enemies","last nav area","last pos x","last pos y","last pos z","last nav 0 friends","last nav 0 enemies","last nav 1 friends","last nav 1 enemies","last nav 2 friends","last nav 2 enemies","last nav 3 friends","last nav 3 enemies","last nav 4 friends","last nav 4 enemies","last nav 5 friends","last nav 5 enemies","last nav 6 friends","last nav 6 enemies","last nav 7 friends","last nav 7 enemies","last nav 8 friends","last nav 8 enemies","last nav 9 friends","last nav 9 enemies","last nav 10 friends","last nav 10 enemies","old nav area","old pos x","old pos y","old pos z","old nav 0 friends","old nav 0 enemies","old nav 1 friends","old nav 1 enemies","old nav 2 friends","old nav 2 enemies","old nav 3 friends","old nav 3 enemies","old nav 4 friends","old nav 4 enemies","old nav 5 friends","old nav 5 enemies","old nav 6 friends","old nav 6 enemies","old nav 7 friends","old nav 7 enemies","old nav 8 friends","old nav 8 enemies","old nav 9 friends","old nav 9 enemies","old nav 10 friends","old nav 10 enemies","delta x","delta y","shoot next","crouch next","nav target"};
    //vector<long> b = {1,30,0,1,0,7,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,7,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,7,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,7};
    std::unique_lock<std::mutex> lk(pythonPlanLock);
    toSendIndexToCSKnowId[stateToSendToPython.tickId.size()] = csknowId;
    stateToSendToPython.tickId.push_back(0);
    stateToSendToPython.roundId.push_back(curState.roundNumber);
    stateToSendToPython.sourcePlayerId.push_back(0);
    stateToSendToPython.sourcePlayerName.push_back(curState.clients[csknowId].name);
    stateToSendToPython.demoName.push_back("live");
    stateToSendToPython.curState.push_back(serverStateToTimeStepState(csknowId, curState));
    stateToSendToPython.lastState.push_back(serverStateToTimeStepState(csknowId, lastState));
    stateToSendToPython.oldState.push_back(serverStateToTimeStepState(csknowId, oldState));
    stateToSendToPython.plan.push_back({});

    // wait for main thread to handle communication with python
    // need to check if got this threads results as could write while a request is outstanding, those results come back
    // and keep waiting for next request to happen
    pythonPlanCV.wait(lk, [&] { return csknowIdToReceivedState.find(csknowId) != csknowIdToReceivedState.end(); });

    int64_t result = csknowIdToReceivedState[csknowId];

    lk.unlock();

    return result;
}

void PythonModelInterface::CommunicateWithPython() {
    // no pipeline, only allow one outstanding request to python at a time
    if (waitingOnPython) {
        string pythonToCppFileName = "python_to_cpp.csv";
        string pythonToCppFilePath = dataPath + "/" + pythonToCppFileName;
        string tmpPythonToCppFileName = "python_to_cpp.csv.tmp.read";
        string tmpPythonToCppFilePath = dataPath + "/" + tmpPythonToCppFileName;

        std::unique_lock<std::mutex> lk(pythonPlanLock);
        if (std::filesystem::exists(pythonToCppFilePath)) {
            std::filesystem::rename(pythonToCppFilePath, tmpPythonToCppFilePath);
            std::ifstream f(tmpPythonToCppFilePath);
            std::string buffer;
            std::getline(f, buffer);
            const char * bufferPtr = buffer.c_str();
            for (size_t curStart = 0, curDelimiter = getNextDelimiter(bufferPtr, curStart, buffer.size()), colNumber = 0;
                 curDelimiter < buffer.size();
                 curStart = curDelimiter + 1, curDelimiter = getNextDelimiter(bufferPtr, curStart, buffer.size()), colNumber++) {
                assert(colNumber < sentIndexToCSKnowId.size());
                // assuming results per player come back in same order as rows sent to python
                readCol(bufferPtr, curStart, curDelimiter, 0, colNumber, csknowIdToReceivedState[sentIndexToCSKnowId[colNumber]]);
            }
            f.close();
            sentIndexToCSKnowId.clear();
            waitingOnPython = false;
        }
        lk.unlock();
        pythonPlanCV.notify_all();
    }
    // recheck as might've received results in this call from python
    if (!waitingOnPython) {
        string cppToPythonFileName = "cpp_to_python.csv";
        string cppToPythonFilePath = dataPath + "/" + cppToPythonFileName;
        string tmpCppToPythonFileName = "cpp_to_python.csv.tmp.read";
        string tmpCppToPythonFilePath = dataPath + "/" + tmpCppToPythonFileName;

        std::unique_lock<std::mutex> lk(pythonPlanLock);

        if (!stateToSendToPython.tickId.empty()) {
            std::ofstream f(tmpCppToPythonFilePath);
            f << stateToSendToPython.toCSV();
            f.close();
            waitingOnPython = true;
            std::filesystem::rename(tmpCppToPythonFilePath, cppToPythonFilePath);
            sentIndexToCSKnowId = toSendIndexToCSKnowId;
        }

        lk.unlock();
    }
}
