//
// Created by durst on 1/17/23.
//

#include "bots/streaming_moments/streaming_test_logger.h"

namespace csknow::test_log {
    void StreamingTestLogger::createLogFiles() {
        testTimingFile.close();
        testTimingFile.open(testTimingPath, std::fstream::out);
        testEventTimingFile.close();
        testEventTimingFile.open(testEventTimingPath, std::fstream::out);
        testTimingFile << "test name,test id,end time" << std::endl;
        testEventTimingFile << "event name,test id,event id,event time" << std::endl;
    }

    void StreamingTestLogger::startTest(const string & testName) {
        if (!testTimings.isEmpty() && testTimings.fromNewest().endTime == defaultTime) {
            throw std::logic_error("starting test while prior one not ended");
        }
        testTimings.enqueue({testName, nextTestId,
                             std::chrono::system_clock::now(), defaultTime,
                             false, {}});
        nextTestId++;
    }

    void StreamingTestLogger::addEvent(const string & eventName, const string & payload) {
        if (testTimings.isEmpty()) {
            throw std::logic_error("adding event to test that isn't started");
        }
        TestTiming & lastTest = testTimings.fromNewest();

        TestId newEventId;
        if (lastTest.testEventTimings.empty()) {
            newEventId = 0;
        }
        else {
            newEventId = lastTest.testEventTimings.back().eventId + 1;
        }

        lastTest.testEventTimings.push_back({eventName, lastTest.testId, newEventId, curFrameTime,
                                             payload});
    }

    void StreamingTestLogger::endTest(bool success) {
        if (testTimings.isEmpty()) {
            throw std::logic_error("ending test without starting one");
        }

        TestTiming & lastTest = testTimings.fromNewest();
        lastTest.endTime = std::chrono::system_clock::now();
        lastTest.success = success;
        testTimingFile << lastTest.toString() << std::endl;
        //testTimingFile.flush();
        for (const auto & testEventTiming : lastTest.testEventTimings) {
            testEventTimingFile << testEventTiming.toString(lastTest.startTime) << std::endl;
        }
        //testEventTimingFile.flush();
    }

    bool StreamingTestLogger::testActive() const {
        return !testTimings.isEmpty() && testTimings.fromNewest().endTime == defaultTime;
    }
}
