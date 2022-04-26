//
// Created by durst on 4/26/22.
//

#ifndef CSKNOW_NODES_H
#define CSKNOW_NODES_H

#include "load_save_bot_data.h"

class Node {
public:
    virtual bool relevant(const ServerState & state)
    virtual void onEntry()
};

class RootNode {

};

#endif //CSKNOW_NODES_H
