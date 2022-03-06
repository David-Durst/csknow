//
// Created by durst on 3/1/22.
//
#ifndef CSKNOW_MODEL_PLAN_H
#define CSKNOW_MODEL_PLAN_H
#include "load_save_bot_data.h"
#include "queries/train_dataset.h"
#include "navmesh/nav_file.h"
#include <optional>
#include <mutex>
#include <condition_variable>
#include <thread>


// state for communicating with python model - one will be made in manage_thinkers, will be referenced here
class PythonModelInterface {
    std::mutex pythonPlanLock;
    std::condition_variable pythonPlanCV;
    TrainDatasetResult stateToSendToPython;
    map<size_t, int32_t> toSendIndexToCSKnowId;
    // once sent, copy over here so that can build up next toSend list while waiting for pythong to respond
    map<size_t, int32_t> sentIndexToCSKnowId;
    // if a thread stalls for a while, writing results by id ensures it's state will be around when it wakes up
    map<int32_t, int64_t> csknowIdToReceivedState;
    nav_mesh::nav_file navFile;
    string dataPath;

    // variable to check if waiting for results to come back
    // if false, free to send new states to Python
    bool waitingOnPython;

    TrainDatasetResult::TimeStepState serverStateToTimeStepState(int32_t csknowId, ServerState serverState);

public:
    // the planning thread calls this, blocks on adding state to stateToSnedToPython and is only woken up
    // once the main thread has written that state to disk, python has read it and made prediction, and
    // main thread has read prediction back and added it to resultsFromPythong
    int64_t GetTargetNavArea(int32_t csknowId, ServerState curState, ServerState lastState, ServerState oldState);

    // the main thread calls this, handles communication with python if (a) waiting for results from python and
    // python finished or (b) not waiting on results and a planning thread wants a prediction from python
    // THIS DOESN'T BLOCK FOR ANOTHER THREAD TO NOTIFY ON A CV, JUST BLOCKS ON pythonPlanLock
    void CommunicateWithPython();

    void reloadNavFile(string navPath) { navFile.load(navPath); }

    PythonModelInterface(string dataPath) : dataPath(dataPath) { };
};

#endif //CSKNOW_MODEL_PLAN_H
