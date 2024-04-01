//
// Created by durst on 4/1/24.
//

#include "bots/behavior_tree/node.h"

std::optional<int> _scriptIndex = std::nullopt;

void setScriptRestart(std::optional<int> scriptIndex) {
    _scriptIndex = scriptIndex;
}

std::optional<int> getScriptRestart() {
    return _scriptIndex;
}
