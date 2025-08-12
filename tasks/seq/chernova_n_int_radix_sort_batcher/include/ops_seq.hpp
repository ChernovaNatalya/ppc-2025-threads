#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace chernova_n_int_radix_sort_batcher_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  unsigned int input_size, output_size;
  const size_t base_size = 3;

  void MergeSortBatcher(std::vector<int>& arr, std::vector<int>& buf, size_t base_size);
  void ComparePairs(std::vector<int>& arr);
  void Merge(std::vector<int>& arr, std::vector<int>& buf, size_t left_len, size_t right_len, int odd);
  void RadixSort(std::vector<int>& data);
};

}  // namespace chernova_n_int_radix_sort_batcher_seq