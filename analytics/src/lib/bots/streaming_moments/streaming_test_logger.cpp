//
// Created by durst on 1/17/23.
//

#include "bots/streaming_moments/streaming_test_logger.h"

namespace csknow::test_log {
    void StreamingTestLogger::startTest(const string & testName) {
        TestId newId;
        if (testTimings.isEmpty()) {
            newId = 0;
        }
        else {
            if (testTimings.fromNewest().endTime == defaultTime) {
                throw std::logic_error("starting test while prior one not ended");
            }
            else {
                newId = testTimings.fromNewest().testId + 1;
            }
        }
        testTimings.enqueue({testName, newId, std::chrono::system_clock::now(), defaultTime});
    }

    void StreamingTestLogger::addEvent(const string & eventName) {
        if (testTimings.isEmpty()) {
            throw std::logic_error("adding event to test that isn't started");
        }
        TestTiming & lastTest = testTimings.fromNewest();

        TestId newEventId;
        if (lastTest.testEventTimings.empty()) {
            newEventId = 0;
        }
        else {
            newEventId = lastTest.testEventTimings.back().eventId;
        }

        lastTest.testEventTimings.push_back({eventName, lastTest.testId, newEventId,
                                             std::chrono::system_clock::now()});
    }

    void StreamingTestLogger::endTest(bool success) {
        if (testTimings.isEmpty()) {
            throw std::logic_error("ending test without starting one");
        }

        TestTiming & lastTest = testTimings.fromNewest();
        lastTest.endTime = std::chrono::system_clock::now();
        lastTest.success = success;
        testTimingFile << lastTest.toString() << std::endl;
        testTimingFile.flush();
        for (const auto & testEventTiming : lastTest.testEventTimings) {
            testEventTimingFile << testEventTiming.toString() << std::endl;
        }
        testEventTimingFile.flush();
    }
}
