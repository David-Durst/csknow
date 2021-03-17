#ifndef CSKNOW_QUERY_H
#define CSKNOW_QUERY_H
#include <vector>
#include <sstream>
using std::vector;
using std::stringstream;


class QueryResult {
public:
    virtual string toCSV() = 0;
};

class PredicateResult : public QueryResult {
public:
    vector<int64_t> positionIndex;
    vector<string> demoFile;

    void collectResults(vector<int64_t> * tmpIndices, int numThreads) {
        for (int i = 0; i < numThreads; i++) {
            for (const auto & elem : tmpIndices[i]) {
                positionIndex.push_back(elem);
            }
        }
    }

    virtual string toCSV() {
        stringstream ss;
        ss << "position index" << std::endl;
        for (const auto & index : positionIndex) {
            ss << index << std::endl;
        }
        return ss.str();
    };
};

#endif //CSKNOW_QUERY_H
