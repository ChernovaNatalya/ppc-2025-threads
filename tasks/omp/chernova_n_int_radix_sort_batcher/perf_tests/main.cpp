#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/chernova_n_int_radix_sort_batcher/include/ops_omp.hpp"

namespace {

void SetupTaskData(std::vector<int> &in, std::vector<int> &out, std::shared_ptr<ppc::core::TaskData> &task_data) {
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());
}

void Execution(auto &test_task) {
  ASSERT_EQ(test_task.Validation(), true);
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();
}

bool CheckResult(std::vector<int> &result) {
  for (long unsigned int i = 0; i < result.size() - 2; i++) {
    if (result[i] > result[i + 1]) {
      return false;
    }
  }
  return true;
}

void GenerateRandomNumbers(size_t count, std::vector<int> &output, int max_value) {
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<int> dist(-max_value, max_value);

  for (size_t i = 0; i < count; ++i) {
    output.push_back(dist(gen));
  }
}

}  // namespace

TEST(chernova_n_int_radix_sort_batcher_omp, test_pipeline_run) {
  const int k = 1000000;
  std::vector<int> in;
  GenerateRandomNumbers(k, in, 10000000);
  std::vector<int> out(k, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  SetupTaskData(in, out, task_data);

  // Create Task
  auto test_task = std::make_shared<chernova_n_int_radix_sort_batcher_omp::TestTaskOpenMP>(task_data);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  EXPECT_TRUE(CheckResult(out));
}

TEST(chernova_n_int_radix_sort_batcher_omp, test_task_run) {
  const int k = 1000000;
  std::vector<int> in;
  GenerateRandomNumbers(k, in, 10000000);
  std::vector<int> out(k, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  SetupTaskData(in, out, task_data);

  // Create Task
  auto test_task = std::make_shared<chernova_n_int_radix_sort_batcher_omp::TestTaskOpenMP>(task_data);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  EXPECT_TRUE(CheckResult(out));
}
