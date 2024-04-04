//
// Created by durst on 4/4/24.
//

#include "bots/testing/log_helpers.h"
#include <chrono>
#include <sstream>
#include <iomanip>

std::string getNowAsISOString() {
    std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream ss;
    ss << std::put_time( std::localtime( &t ), "%FT%T%z");
    return ss.str();
}

std::string getNowAsFileNameString() {
    std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream ss;
    ss << std::put_time( std::localtime( &t ), "%Y_%m_%d_%H_%M_%S");
    return ss.str();
}
