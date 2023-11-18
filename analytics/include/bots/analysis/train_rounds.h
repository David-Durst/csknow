//
// Created by durst on 11/17/23.
//

#ifndef CSKNOW_TRAIN_ROUNDS_H
#define CSKNOW_TRAIN_ROUNDS_H

#include <map>
#include <string>
#include <vector>
#include <set>
#include <filesystem>

std::map<std::string, std::map<std::string, std::vector<int64_t>>>
load_groups_to_train_round_ids(const std::filesystem::path & dir);

std::set<int64_t>
load_one_data_file_train_round_ids(const std::filesystem::path & trainTestSplitDir, const std::string & dataFileStr);

#endif //CSKNOW_TRAIN_ROUNDS_H
