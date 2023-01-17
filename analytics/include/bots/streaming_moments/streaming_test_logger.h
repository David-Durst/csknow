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

        string toString() const {
            std::stringstream ss;
            ss << eventName << "," << testId << "," << eventId
               << "," << std::chrono::duration_cast<std::chrono::seconds>(eventTime.time_since_epoch()).count();
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
                << "," << std::chrono::duration_cast<std::chrono::seconds>(startTime.time_since_epoch()).count()
                << "," << std::chrono::duration_cast<std::chrono::seconds>(endTime.time_since_epoch()).count()
                << "," << boolToInt(success);
            return ss.str();
        }
    };

    class StreamingTestLogger {
        CircularBuffer<TestTiming> testTimings;
        std::fstream testTimingFile, testEventTimingFile;
        std::fstream eventsTimingFile;

    public:
        StreamingTestLogger(const string & navPath) : testTimings(STREAMING_HISTORY_TICKS),
            testTimingFile(fs::path(navPath) / fs::path("..") / fs::path("..") /
                fs::path("test_timing.csv"), std::fstream::out),
            testEventTimingFile(fs::path(navPath) / fs::path("..") / fs::path("..") /
                fs::path("test_event_timing.csv"), std::fstream::out) {
            testTimingFile << "test name,test id,start time,end time" << std::endl;
            testEventTimingFile << "event name,test id,event id,event time" << std::endl;
        }

        void startTest(const string & testName);
        void addEvent(const string & eventName);
        void endTest(bool success);
        bool testActive();
    };
}
#endif //CSKNOW_STREAMING_TEST_LOGGER_H
