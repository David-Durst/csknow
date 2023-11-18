//
// Created by durst on 11/17/23.
//

#include <bots/analysis/train_rounds.h>
#include <set>
#include "highfive/H5File.hpp"

std::string trainRoundIdsFileName = "train_round_ids.hdf5";


std::map<std::string, std::map<std::string, std::vector<int64_t>>>
load_groups_to_train_round_ids(const std::filesystem::path & dir) {
    std::map<std::string, std::map<std::string, std::vector<int64_t>>> result;

    HighFive::File file(dir / trainRoundIdsFileName, HighFive::File::ReadOnly);

    for (size_t groupIndex = 0; groupIndex < file.getNumberObjects(); groupIndex++) {
        std::string groupName = file.getObjectName(groupIndex);
        if (file.getObjectType(groupName) != HighFive::ObjectType::Group) {
            continue;
        }
        HighFive::Group group = file.getGroup(groupName);
        for (size_t datasetIndex = 0; datasetIndex < group.getNumberObjects(); datasetIndex++) {
            std::string datasetName = group.getObjectName(datasetIndex);
            if (group.getObjectType(datasetName) != HighFive::ObjectType::Dataset) {
                continue;
            }
            result[groupName][datasetName] = group.getDataSet(datasetName).read<std::vector<int64_t>>();
        }
    }

    return result;
}

std::set<int64_t>
load_one_data_file_train_round_ids(const std::filesystem::path &trainTestSplitDir, const std::string &dataFileStr) {
    std::map<std::string, std::map<std::string, std::vector<int64_t>>> groupsToTrainRoundIds =
            load_groups_to_train_round_ids(trainTestSplitDir);
    std::filesystem::path dataFilePath(dataFileStr);
    std::vector<int64_t> result_vector =
            groupsToTrainRoundIds[dataFilePath.parent_path().filename()][dataFilePath.filename()];
    return {result_vector.begin(), result_vector.end()};
}