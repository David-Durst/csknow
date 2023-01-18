//
// Created by durst on 1/17/23.
//

#ifndef CSKNOW_STREAMING_TEST_LOGGER_H
#define CSKNOW_STREAMING_TEST_LOGGER_H

#include "bots/streaming_bot_database.h"
#include <filesystem>
namespace fs = std::filesystem;

typedef int64_t TestId;

namespace csknow::test_log {
    struct TestEventTiming {
        string eventName;
        TestId testId, eventId;
        CSKnowTime eventTime;
        string payload;

        string toString(const CSKnowTime & testStartTime) const {
            std::stringstream ss;
            ss << eventName << "," << testId << "," << eventId
               << "," << std::chrono::duration<double>(eventTime - testStartTime).count()
               << "," << payload;
            return ss.str();
        }
    };

    struct TestTiming {
        string testName;
        TestId testId;
        CSKnowTime startTime = defaultTime, endTime = defaultTime;
        bool success;
        vector<TestEventTiming> testEventTimings;

        string toString() const {
            std::stringstream ss;
            ss << testName << "," << testId
                << "," << std::chrono::duration<double>(endTime - startTime).count()
                << "," << boolToInt(success);
            return ss.str();
        }
    };

    class StreamingTestLogger {
        CircularBuffer<TestTiming> testTimings;
        fs::path testTimingPath, testEventTimingPath;
        std::fstream testTimingFile, testEventTimingFile;
        int nextTestId = 0;
        // ensures multiple events on same frame get same time
        CSKnowTime curFrameTime = defaultTime;

    public:
        CSGOId attackerId = INVALID_ID;

        explicit StreamingTestLogger(const string & navPath) : testTimings(STREAMING_HISTORY_TICKS),
            testTimingPath(fs::path(navPath) / fs::path("..") / fs::path("..") / fs::path("test_timing.csv")),
            testEventTimingPath(fs::path(navPath) / fs::path("..") / fs::path("..") / fs::path("test_event_timing.csv")),
            // open these for append since blackboard recreates them so often
            // createLogFiles will reopen with concate for first test
            testTimingFile(testTimingPath, std::fstream::out | std::fstream::app),
            testEventTimingFile(testEventTimingPath, std::fstream::out | std::fstream::app) { }

        void createLogFiles();
        void setNextTestId(int nextTestId) { this->nextTestId = nextTestId; }
        void setCurFrameTime() { curFrameTime = std::chrono::system_clock::now(); }
        void startTest(const string & testName);
        void addEvent(const string & eventName, const string & payload);
        void endTest(bool success);
        bool testActive() const;
    };
}
#endif //CSKNOW_STREAMING_TEST_LOGGER_H
