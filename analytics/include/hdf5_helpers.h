//
// Created by durst on 5/16/23.
//

#ifndef CSKNOW_HDF5_HELPERS_H
#define CSKNOW_HDF5_HELPERS_H

#include <array>
#include <vector>
#include <sstream>
#include <string>
#include <fstream>
#include <functional>
#include <highfive/H5File.hpp>
#include "linear_algebra.h"
#include "enum_helpers.h"
using std::array;
using std::string;

static inline
string toSignedIntString(int64_t i, bool dropZero = false) {
    if (dropZero && i == 0) {
        return "";
    }
    if (i < 0) {
        return std::to_string(i);
    }
    else {
        return "+" + std::to_string(i);

    }
}


template <typename T>
std::vector<int> vectorOfEnumsToVectorOfInts(const std::vector<T> & vectorOfEnums) {
    std::vector<int> result;
    result.reserve(vectorOfEnums.size());

    for (size_t i = 0; i < vectorOfEnums.size(); i++) {
        result.push_back(enumAsInt(vectorOfEnums[i]));
    }

    return result;
}

template <typename T>
void loadVectorOfEnums(HighFive::File & file, string datasetName, std::vector<T> & vectorOfEnums) {
    auto intDataset = file.getDataSet(datasetName).read<std::vector<int>>();
    vectorOfEnums.reserve(intDataset.size());

    for (const auto & element : intDataset) {
        vectorOfEnums.push_back(static_cast<T>(element));
    }
}

template <typename T>
std::vector<T> vectorOfVectorToVectorSelector(const std::vector<std::vector<T>> & vectorOfVector, size_t index) {
    std::vector<T> result;
    result.reserve(vectorOfVector.size());

    for (size_t i = 0; i < vectorOfVector.size(); i++) {
        result.push_back(vectorOfVector[i][index]);
    }

    return result;
}

template <typename T, std::size_t N>
std::vector<array<int, N>> vectorOfEnumArraysToVectorOfIntArrays(const std::vector<std::array<T, N>> & vectorOfArrays) {
    std::vector<std::array<int, N>> result;
    result.reserve(vectorOfArrays.size());

    for (size_t vectorIndex = 0; vectorIndex < vectorOfArrays.size(); vectorIndex++) {
        result.push_back({});
        for (size_t arrayIndex = 0; arrayIndex < vectorOfArrays.front().size(); arrayIndex++) {
            result[vectorIndex][arrayIndex] = enumAsInt(vectorOfArrays[vectorIndex][arrayIndex]);
        }
    }

    return result;
}

template <std::size_t N>
std::array<std::vector<std::array<double, N>>, 2>
vectorOfVec2ArraysToArrayOfDoubleVectors(const std::vector<std::array<Vec2, N>> & vectorOfVec2Arrays){
    std::array<std::vector<std::array<double, N>>, 2> result;
    for (size_t i = 0; i < result.size(); i++) {
        result[0].reserve(vectorOfVec2Arrays.size());
        result[1].reserve(vectorOfVec2Arrays.size());
    }

    for (size_t vectorIndex = 0; vectorIndex < vectorOfVec2Arrays.size(); vectorIndex++) {
        result[0].push_back({});
        result[1].push_back({});
        for (size_t arrayIndex = 0; arrayIndex < vectorOfVec2Arrays.front().size(); arrayIndex++) {
            result[0][vectorIndex][arrayIndex] = vectorOfVec2Arrays[vectorIndex][arrayIndex].x;
            result[1][vectorIndex][arrayIndex] = vectorOfVec2Arrays[vectorIndex][arrayIndex].y;
        }
    }

    return result;
}

template <std::size_t N>
std::array<std::vector<std::array<double, N>>, 3>
vectorOfVec3ArraysToArrayOfDoubleVectors(const std::vector<std::array<Vec3, N>> & vectorOfVec3Arrays){
    std::array<std::vector<std::array<double, N>>, 3> result;
    for (size_t i = 0; i < result.size(); i++) {
        result[0].reserve(vectorOfVec3Arrays.size());
        result[1].reserve(vectorOfVec3Arrays.size());
        result[2].reserve(vectorOfVec3Arrays.size());
    }

    for (size_t vectorIndex = 0; vectorIndex < vectorOfVec3Arrays.size(); vectorIndex++) {
        result[0].push_back({});
        result[1].push_back({});
        result[2].push_back({});
        for (size_t arrayIndex = 0; arrayIndex < vectorOfVec3Arrays.front().size(); arrayIndex++) {
            result[0][vectorIndex][arrayIndex] = vectorOfVec3Arrays[vectorIndex][arrayIndex].x;
            result[1][vectorIndex][arrayIndex] = vectorOfVec3Arrays[vectorIndex][arrayIndex].y;
            result[2][vectorIndex][arrayIndex] = vectorOfVec3Arrays[vectorIndex][arrayIndex].z;
        }
    }

    return result;
}

template <typename T, std::size_t N>
void saveTemporalVectorOfEnumsToHDF5(const std::vector<std::array<T, N>> & vectorOfEnumArrays, HighFive::File & file,
                                     int startOffset, const string & baseString,
                                     const HighFive::DataSetCreateProps & hdf5CreateProps) {
    std::vector<std::array<int, N>> vectorOfIntArrays = vectorOfEnumArraysToVectorOfIntArrays(vectorOfEnumArrays);
    HighFive::DataSet dataset = file.createDataSet("/data/" + baseString, vectorOfIntArrays, hdf5CreateProps);
    HighFive::Attribute names = dataset.createAttribute<std::string>("column names",
                                                                     HighFive::DataSpace::From(baseString));
    std::stringstream namesStream;
    for (size_t arrayIndex = 0; arrayIndex < vectorOfEnumArrays.front().size(); arrayIndex++) {
        if (arrayIndex > 0) {
            namesStream << ",";
        }
        namesStream << baseString + " (t" + toSignedIntString(arrayIndex + startOffset, true) + ")";
    }
    names.write(namesStream.str());
}

template <typename T, std::size_t N>
void saveArrayOfVectorsToHDF5(const std::array<std::vector<T>, N> & arrayOfVectors, HighFive::File & file,
                              std::string baseString, const std::vector<string> columnNames,
                              const HighFive::DataSetCreateProps & hdf5CreateProps) {
    HighFive::DataSet dataset = file.createDataSet("/data/" + baseString, arrayOfVectors, hdf5CreateProps);
    HighFive::Attribute names = dataset.createAttribute<std::string>("column names",
                                                                     HighFive::DataSpace::From(baseString));
    std::stringstream namesStream;
    for (size_t arrayIndex = 0; arrayIndex < columnNames.size(); arrayIndex++) {
        if (arrayIndex > 0) {
            namesStream << ",";
        }
        namesStream << columnNames[arrayIndex];
    }
    names.write(namesStream.str());
}

template <typename T, std::size_t N>
void saveTemporalArrayOfVectorsToHDF5(const std::vector<std::array<T, N>> & vectorOfArrays, HighFive::File & file,
                                      int startOffset, const string & baseString,
                                      const HighFive::DataSetCreateProps & hdf5CreateProps) {
    HighFive::DataSet dataset = file.createDataSet("/data/" + baseString, vectorOfArrays, hdf5CreateProps);
    HighFive::Attribute names = dataset.createAttribute<std::string>("column names",
                                                                     HighFive::DataSpace::From(baseString));
    std::stringstream namesStream;
    for (size_t arrayIndex = 0; arrayIndex < vectorOfArrays.front().size(); arrayIndex++) {
        if (arrayIndex > 0) {
            namesStream << ",";
        }
        namesStream << baseString + " (t" + toSignedIntString(arrayIndex + startOffset, true) + ")";
    }
    names.write(namesStream.str());
}

template <std::size_t N>
void saveTemporalArrayOfVec2VectorsToHDF5(const std::vector<std::array<Vec2, N>> & vectorOfVec2Arrays, HighFive::File & file,
                                          int startOffset, const string & baseString,
                                          const HighFive::DataSetCreateProps & hdf5CreateProps) {
    std::array<std::vector<std::array<double, N>>, 2> arrayOfArrayOfVectors =
                                                           vectorOfVec2ArraysToArrayOfDoubleVectors(vectorOfVec2Arrays);
    HighFive::DataSet xDataset =
            file.createDataSet("/data/" + baseString + " x", arrayOfArrayOfVectors[0], hdf5CreateProps);
    HighFive::DataSet yDataset =
            file.createDataSet("/data/" + baseString + " y", arrayOfArrayOfVectors[1], hdf5CreateProps);

    HighFive::Attribute xNames = xDataset.createAttribute<std::string>("column names",
                                                                       HighFive::DataSpace::From(baseString));
    HighFive::Attribute yNames = yDataset.createAttribute<std::string>("column names",
                                                                       HighFive::DataSpace::From(baseString));

    std::stringstream xNamesStream, yNamesStream;
    for (size_t arrayIndex = 0; arrayIndex < vectorOfVec2Arrays.front().size(); arrayIndex++) {
        if (arrayIndex > 0) {
            xNamesStream << ",";
            yNamesStream << ",";
        }
        xNamesStream << baseString + " x (t" + toSignedIntString(arrayIndex + startOffset, true) + ")";
        yNamesStream << baseString + " y (t" + toSignedIntString(arrayIndex + startOffset, true) + ")";
    }
    xNames.write(xNamesStream.str());
    yNames.write(yNamesStream.str());
}

template <std::size_t N>
void saveTemporalArrayOfVec3VectorsToHDF5(const std::vector<std::array<Vec3, N>> & vectorOfVec3Arrays, HighFive::File & file,
                                          int startOffset, const string & baseString,
                                          const HighFive::DataSetCreateProps & hdf5CreateProps) {
    std::array<std::vector<std::array<double, N>>, 3> arrayOfArrayOfVectors =
                                                           vectorOfVec3ArraysToArrayOfDoubleVectors(vectorOfVec3Arrays);
    HighFive::DataSet xDataset =
            file.createDataSet("/data/" + baseString + " x", arrayOfArrayOfVectors[0], hdf5CreateProps);
    HighFive::DataSet yDataset =
            file.createDataSet("/data/" + baseString + " y", arrayOfArrayOfVectors[1], hdf5CreateProps);
    HighFive::DataSet zDataset =
            file.createDataSet("/data/" + baseString + " z", arrayOfArrayOfVectors[2], hdf5CreateProps);

    HighFive::Attribute xNames = xDataset.createAttribute<std::string>("column names",
                                                                       HighFive::DataSpace::From(baseString));
    HighFive::Attribute yNames = yDataset.createAttribute<std::string>("column names",
                                                                       HighFive::DataSpace::From(baseString));
    HighFive::Attribute zNames = zDataset.createAttribute<std::string>("column names",
                                                                       HighFive::DataSpace::From(baseString));

    std::stringstream xNamesStream, yNamesStream, zNamesStream;
    for (size_t arrayIndex = 0; arrayIndex < vectorOfVec3Arrays.front().size(); arrayIndex++) {
        if (arrayIndex > 0) {
            xNamesStream << ",";
            yNamesStream << ",";
            zNamesStream << ",";
        }
        xNamesStream << baseString + " x (t" + toSignedIntString(arrayIndex + startOffset, true) + ")";
        yNamesStream << baseString + " y (t" + toSignedIntString(arrayIndex + startOffset, true) + ")";
        zNamesStream << baseString + " z (t" + toSignedIntString(arrayIndex + startOffset, true) + ")";
    }
    xNames.write(xNamesStream.str());
    yNames.write(yNamesStream.str());
    zNames.write(zNamesStream.str());
}

#endif //CSKNOW_HDF5_HELPERS_H
