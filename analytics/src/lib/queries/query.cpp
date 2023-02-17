//
// Created by durst on 9/11/22.
//

#include "queries/query.h"
#include "enum_helpers.h"

template <typename T, std::size_t N>
std::vector<int> vectorOfEnumsToVectorOfInts(const std::vector<T> & vectorOfEnums) {
    std::vector<int> result;
    result.reserve(vectorOfEnums.size());

    for (size_t i = 0; i < vectorOfEnums.size(); i++) {
        result.push_back(enumAsInt(vectorOfEnums[i]));
    }

    return result;
}

template <typename T, std::size_t _Nm>
std::array<std::vector<T>, _Nm> vectorOfArraysToArrayOfVectors(const std::vector<std::array<T, _Nm>> & vectorOfArrays) {
    std::array<std::vector<T>, _Nm> arrayOfVectors;
    for (size_t i = 0; i < arrayOfVectors.size(); i++) {
        arrayOfVectors[i].reserve(vectorOfArrays.size());
    }

    for (size_t vectorIndex = 0; vectorIndex < vectorOfArrays.size(); vectorIndex++) {
        for (size_t arrayIndex = 0; arrayIndex < arrayOfVectors.size(); arrayIndex++) {
            arrayOfVectors[arrayIndex].push_back(vectorOfArrays[vectorIndex][arrayIndex]);
        }
    }

    return arrayOfVectors;
}

template <std::size_t _Nm>
std::array<std::array<std::vector<double>, 2>, _Nm>
vectorOfVec2ArraysToArrayOfArrayOfVectors(const std::vector<std::array<Vec2, _Nm>> & vectorOfVec2Arrays){
    std::array<std::array<std::vector<double>, 2>, _Nm> arrayOfArrayOfVectors;
    for (size_t i = 0; i < arrayOfArrayOfVectors.size(); i++) {
        for (size_t j = 0; j < 2; j++) {
            arrayOfArrayOfVectors[i][j].reserve(vectorOfVec2Arrays.size());
        }
    }

    for (size_t vectorIndex = 0; vectorIndex < vectorOfVec2Arrays.size(); vectorIndex++) {
        for (size_t arrayIndex = 0; arrayIndex < arrayOfArrayOfVectors.size(); arrayIndex++) {
            arrayOfArrayOfVectors[arrayIndex][0].push_back(vectorOfVec2Arrays[vectorIndex][arrayIndex].x);
            arrayOfArrayOfVectors[arrayIndex][1].push_back(vectorOfVec2Arrays[vectorIndex][arrayIndex].y);
        }
    }

    return arrayOfArrayOfVectors;
}

template <std::size_t _Nm>
std::array<std::array<std::vector<double>, 3>, _Nm>
vectorOfVec3ArraysToArrayOfArrayOfVectors(const std::vector<std::array<Vec3, _Nm>> & vectorOfVec3Arrays){
    std::array<std::array<std::vector<double>, 3>, _Nm> arrayOfArrayOfVectors;
    for (size_t i = 0; i < arrayOfArrayOfVectors.size(); i++) {
        for (size_t j = 0; j < 3; j++) {
            arrayOfArrayOfVectors[i][j].reserve(vectorOfVec3Arrays.size());
        }
    }

    for (size_t vectorIndex = 0; vectorIndex < vectorOfVec3Arrays.size(); vectorIndex++) {
        for (size_t arrayIndex = 0; arrayIndex < arrayOfArrayOfVectors.size(); arrayIndex++) {
            arrayOfArrayOfVectors[arrayIndex][0].push_back(vectorOfVec3Arrays[vectorIndex][arrayIndex].x);
            arrayOfArrayOfVectors[arrayIndex][1].push_back(vectorOfVec3Arrays[vectorIndex][arrayIndex].y);
            arrayOfArrayOfVectors[arrayIndex][2].push_back(vectorOfVec3Arrays[vectorIndex][arrayIndex].z);
        }
    }

    return arrayOfArrayOfVectors;
}

template <typename T>
void saveTemporalVectorOfEnumsToHDF5(const std::vector<T> & vectorOfEnums, HighFive::File & file,
                                     int startOffset, int endOffset, const string & baseString) {
    std::vector<T> vectorOfInts = vectorOfEnumsToVectorOfInts(vectorOfEnums);
    for (int baseZeroOffset = 0; baseZeroOffset < endOffset - startOffset; baseZeroOffset++) {
        H5Easy::dump(file, "/data/" +
                     baseString + "(t" + toSignedIntString(baseZeroOffset + startOffset, true) + ")",
                     vectorOfInts);
    }
}

template <typename T, std::size_t _Nm>
void saveTemporalArrayOfVectorsToHDF5(const std::vector<std::array<T, _Nm>> & vectorOfArrays, HighFive::File & file,
                                      int startOffset, int endOffset, const string & baseString) {
    std::array<std::vector<T>, _Nm> arrayOfVectors = vectorOfArraysToArrayOfVectors(vectorOfArrays);
    for (int baseZeroOffset = 0; baseZeroOffset < endOffset - startOffset; baseZeroOffset++) {
        for (size_t arrayIndex = 0; arrayIndex < arrayOfVectors.size(); arrayIndex++) {
            H5Easy::dump(file, "/data/" +
                         baseString + "(t" + toSignedIntString(baseZeroOffset + startOffset, true) + ")",
                         arrayOfVectors[arrayIndex]);
        }
    }
}

template <typename T, std::size_t _Nm>
void saveTemporalArrayOfVec2VectorsToHDF5(const std::vector<std::array<Vec2, _Nm>> & vectorOfVec2Arrays, HighFive::File & file,
                                          int startOffset, int endOffset, const string & baseString) {
    std::array<std::array<std::vector<T>, 2>, _Nm> arrayOfArrayOfVectors =
            vectorOfVec2ArraysToArrayOfArrayOfVectors(vectorOfVec2Arrays);
    for (int baseZeroOffset = 0; baseZeroOffset < endOffset - startOffset; baseZeroOffset++) {
        for (size_t arrayIndex = 0; arrayIndex < arrayOfArrayOfVectors.size(); arrayIndex++) {
            H5Easy::dump(file, "/data/" +
                         baseString + " x (t" + toSignedIntString(baseZeroOffset + startOffset, true) + ")",
                         arrayOfArrayOfVectors[arrayIndex][0]);
            H5Easy::dump(file, "/data/" +
                         baseString + " y (t" + toSignedIntString(baseZeroOffset + startOffset, true) + ")",
                         arrayOfArrayOfVectors[arrayIndex][1]);
        }
    }
}

template <typename T, std::size_t _Nm>
void saveTemporalArrayOfVec3VectorsToHDF5(const std::vector<std::array<Vec3, _Nm>> & vectorOfVec3Arrays, HighFive::File & file,
                                          int startOffset, int endOffset, const string & baseString) {
    std::array<std::array<std::vector<T>, 2>, _Nm> arrayOfArrayOfVectors =
            vectorOfVec3ArraysToArrayOfArrayOfVectors(vectorOfVec3Arrays);
    for (int baseZeroOffset = 0; baseZeroOffset < endOffset - startOffset; baseZeroOffset++) {
        for (size_t arrayIndex = 0; arrayIndex < arrayOfArrayOfVectors.size(); arrayIndex++) {
            H5Easy::dump(file, "/data/" +
                         baseString + " x (t" + toSignedIntString(baseZeroOffset + startOffset, true) + ")",
                         arrayOfArrayOfVectors[arrayIndex][0]);
            H5Easy::dump(file, "/data/" +
                         baseString + " y (t" + toSignedIntString(baseZeroOffset + startOffset, true) + ")",
                         arrayOfArrayOfVectors[arrayIndex][1]);
            H5Easy::dump(file, "/data/" +
                         baseString + " z (t" + toSignedIntString(baseZeroOffset + startOffset, true) + ")",
                         arrayOfArrayOfVectors[arrayIndex][2]);
        }
    }
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

