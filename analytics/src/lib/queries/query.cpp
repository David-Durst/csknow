//
// Created by durst on 9/11/22.
//

#include "queries/query.h"
#include "enum_helpers.h"

void QueryResult::toHDF5(const string &filePath) {
    // We create an empty HDF55 file, by truncating an existing
    // file if required:
    HighFive::File file(filePath, HighFive::File::Overwrite);

    // create id column
    vector<int64_t> id_to_save;
    id_to_save.reserve(size);
    for (int64_t i = 0; i < size; i++) {
        id_to_save.push_back(i);
    }
    HighFive::DataSetCreateProps hdf5CreateProps;
    hdf5CreateProps.add(HighFive::Deflate(6));
    hdf5CreateProps.add(HighFive::Chunking(id_to_save.size()));
    file.createDataSet(hdf5Prefix + "id", id_to_save, hdf5CreateProps);

    // create all other columns
    toHDF5Inner(file);
}

void saveVec3VectorToHDF5(const std::vector<Vec3> & vectorOfVec3, HighFive::File & file,
                          const string & baseString, const HighFive::DataSetCreateProps & hdf5CreateProps) {
    std::array<std::vector<double>, 3> result;
    for (size_t i = 0; i < result.size(); i++) {
        result[0].reserve(vectorOfVec3.size());
        result[1].reserve(vectorOfVec3.size());
        result[2].reserve(vectorOfVec3.size());
    }

    for (size_t vectorIndex = 0; vectorIndex < vectorOfVec3.size(); vectorIndex++) {
        result[0].push_back(vectorOfVec3[vectorIndex].x);
        result[1].push_back(vectorOfVec3[vectorIndex].y);
        result[2].push_back(vectorOfVec3[vectorIndex].z);
    }
    HighFive::DataSet xDataset =
        file.createDataSet("/data/" + baseString + " x", result[0], hdf5CreateProps);
    HighFive::DataSet yDataset =
        file.createDataSet("/data/" + baseString + " y", result[1], hdf5CreateProps);
    HighFive::DataSet zDataset =
        file.createDataSet("/data/" + baseString + " z", result[2], hdf5CreateProps);
}

void mergeThreadResults(int numThreads, vector<RangeIndexEntry> &rowIndicesPerRound, const vector<vector<int64_t>> & tmpRoundIds,
                        const vector<vector<int64_t>> & tmpRoundStarts, const vector<vector<int64_t>> & tmpRoundSizes,
                        vector<int64_t> & resultStartTickId, int64_t & resultSize,
                        const std::function<void(int64_t, int64_t)> & appendToResult) {
    vector<int64_t> roundsProcessedPerThread(numThreads, 0);
    while (true) {
        bool roundToProcess = false;
        int64_t minThreadId = -1;
        int64_t minRoundId = -1;
        for (int64_t threadId = 0; threadId < numThreads; threadId++) {
            if (roundsProcessedPerThread[threadId] < static_cast<int64_t>(tmpRoundIds[threadId].size())) {
                roundToProcess = true;
                if (minThreadId == -1 || tmpRoundIds[threadId][roundsProcessedPerThread[threadId]] < minRoundId) {
                    minThreadId = threadId;
                    minRoundId = tmpRoundIds[minThreadId][roundsProcessedPerThread[minThreadId]];
                }

            }
        }
        if (!roundToProcess) {
            break;
        }
        rowIndicesPerRound.push_back({});
        rowIndicesPerRound[minRoundId].minId = static_cast<int64_t>(resultStartTickId.size());
        int64_t roundStart = tmpRoundStarts[minThreadId][roundsProcessedPerThread[minThreadId]];
        int64_t roundEnd = roundStart + tmpRoundSizes[minThreadId][roundsProcessedPerThread[minThreadId]];
        for (int64_t tmpRowId = roundStart; tmpRowId < roundEnd; tmpRowId++) {
            appendToResult(minThreadId, tmpRowId);
        }
        rowIndicesPerRound[minRoundId].maxId = static_cast<int64_t>(resultStartTickId.size()) - 1;
        roundsProcessedPerThread[minThreadId]++;
    }
    resultSize = static_cast<int64_t>(resultStartTickId.size());
}

