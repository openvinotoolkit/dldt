// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "timetests_helper/timer.h"
#include <chrono>
#include <fstream>
#include <memory>
#include <string>

#include "statistics_writer.h"

using time_point = std::chrono::high_resolution_clock::time_point;

namespace TimeTest {

int order_count = 0;

Timer::Timer(const std::string &timer_name) {
  name = timer_name;
  order_count++;
  start_time = std::chrono::high_resolution_clock::now();
  StatisticsWriter::Instance().addOrderCount({name, order_count});
}

Timer::~Timer() {
  float duration = std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::high_resolution_clock::now() - start_time)
                       .count();
  StatisticsWriter::Instance().addToTimeStructure({name, duration});
  StatisticsWriter::Instance().deleteOrderCount();
}

} // namespace TimeTest