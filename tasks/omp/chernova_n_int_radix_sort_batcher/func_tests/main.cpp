#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
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

TEST(chernova_n_int_radix_sort_batcher_omp, test_20_numbers) {
  std::vector<int> in{-55, 10, 3, -8, 2, 0, 12, -15, 5, 4, 567, 23, 354, 45, 76, -23, 8, -1, -1, -1};
  std::vector<int> out(20, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  SetupTaskData(in, out, task_data);

  chernova_n_int_radix_sort_batcher_omp::TestTaskOpenMP test_task(task_data);
  Execution(test_task);
  EXPECT_TRUE(CheckResult(out));
}

TEST(chernova_n_int_radix_sort_batcher_omp, test_20_big_numbers) {
  std::vector<int> in{83481,  -37023, 26482,  -82725, -31939, 51613, -10214, 84528,  -97726, 40742,
                      -41172, -60458, -20599, 27106,  64639,  37208, -9964,  -85876, 8627,   86138};
  std::vector<int> out(20, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  SetupTaskData(in, out, task_data);

  chernova_n_int_radix_sort_batcher_omp::TestTaskOpenMP test_task(task_data);
  Execution(test_task);
  EXPECT_TRUE(CheckResult(out));
}

TEST(chernova_n_int_radix_sort_batcher_omp, test_20_random_numbers) {
  const int k = 20;
  std::vector<int> in;
  GenerateRandomNumbers(k, in, 1000);
  std::vector<int> out(k, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  SetupTaskData(in, out, task_data);

  chernova_n_int_radix_sort_batcher_omp::TestTaskOpenMP test_task(task_data);
  Execution(test_task);
  EXPECT_TRUE(CheckResult(out));
}

TEST(chernova_n_int_radix_sort_batcher_omp, test_50_random_numbers) {
  const int k = 50;
  std::vector<int> in;
  GenerateRandomNumbers(k, in, 1000);
  std::vector<int> out(k, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  SetupTaskData(in, out, task_data);

  chernova_n_int_radix_sort_batcher_omp::TestTaskOpenMP test_task(task_data);
  Execution(test_task);
  EXPECT_TRUE(CheckResult(out));
}

TEST(chernova_n_int_radix_sort_batcher_omp, test_1000_numbers) {
  const int k = 1000;
  std::vector<int> in;
  GenerateRandomNumbers(k, in, 1000);
  std::vector<int> out(k, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  SetupTaskData(in, out, task_data);

  chernova_n_int_radix_sort_batcher_omp::TestTaskOpenMP test_task(task_data);
  Execution(test_task);
  EXPECT_TRUE(CheckResult(out));
}

TEST(chernova_n_int_radix_sort_batcher_omp, test_1000_big_numbers) {
  const int k = 1000;
  std::vector<int> in;
  GenerateRandomNumbers(k, in, 100000);
  std::vector<int> out(k, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  SetupTaskData(in, out, task_data);

  chernova_n_int_radix_sort_batcher_omp::TestTaskOpenMP test_task(task_data);
  Execution(test_task);
  EXPECT_TRUE(CheckResult(out));
}
