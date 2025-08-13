#include "omp/chernova_n_int_radix_sort_batcher/include/ops_omp.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

bool chernova_n_int_radix_sort_batcher_omp::TestTaskOpenMP::PreProcessingImpl() {
  input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);
  output_size = task_data->outputs_count[0];

  return true;
}

bool chernova_n_int_radix_sort_batcher_omp::TestTaskOpenMP::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool chernova_n_int_radix_sort_batcher_omp::TestTaskOpenMP::RunImpl() {
  size_t portion = input_size / base_size;
#pragma omp parallel
  {
#pragma omp single
    MergeSortBatcher(input_, portion);
  }
  return true;
}

void chernova_n_int_radix_sort_batcher_omp::TestTaskOpenMP::RadixSort(std::vector<int> &data) {
  size_t len = data.size();
  if (len <= 1) return;

  const int total_bits = sizeof(int) * CHAR_BIT;
  const unsigned int sign_mask = (total_bits > 0) ? (1U << (total_bits - 1)) : 0;
  const int iterations = sizeof(int);

  std::vector<unsigned int> temp(data.size());
  std::vector<unsigned int> output(data.size());

  for (size_t idx = 0; idx < len; ++idx) {
    temp[idx] = static_cast<unsigned int>(data[idx]) ^ sign_mask;
  }

  unsigned int *source = temp.data();
  unsigned int *target = output.data();

  for (int phase = 0; phase < iterations; ++phase) {
    int freq[256] = {0};

    for (size_t i = 0; i < len; ++i) {
      unsigned char chunk = (source[i] >> (phase * 8)) & 0xFF;
      ++freq[chunk];
    }

    for (int i = 1; i < 256; ++i) {
      freq[i] += freq[i - 1];
    }

    for (int i = (int)len - 1; i >= 0; --i) {
      unsigned char chunk = (source[i] >> (phase * 8)) & 0xFF;
      target[freq[chunk] - 1] = source[i];
      --freq[chunk];
    }

    std::swap(source, target);
  }

  for (size_t idx = 0; idx < len; ++idx) {
    data[idx] = static_cast<int>(source[idx] ^ sign_mask);
  }
}

void chernova_n_int_radix_sort_batcher_omp::TestTaskOpenMP::Merge(std::vector<int> &arr, std::vector<int> &buf,
                                                                  size_t left_len, size_t right_len, int odd) {
  for (size_t pos = odd; pos < left_len; pos += 2) buf[pos] = arr[pos];

  size_t left_idx = odd, right_idx = odd, merged_idx = odd;
  while (left_idx < left_len && right_idx < right_len) {
    if (buf[left_idx] <= arr[left_len + right_idx]) {
      arr[merged_idx] = buf[left_idx];
      left_idx += 2;
    } else {
      arr[merged_idx] = arr[left_len + right_idx];
      right_idx += 2;
    }
    merged_idx += 2;
  }

  while (left_idx < left_len) {
    arr[merged_idx] = buf[left_idx];
    left_idx += 2;
    merged_idx += 2;
  }
  while (right_idx < right_len) {
    arr[merged_idx] = arr[left_len + right_idx];
    right_idx += 2;
    merged_idx += 2;
  }
}

void chernova_n_int_radix_sort_batcher_omp::TestTaskOpenMP::ComparePairs(std::vector<int> &arr) {
  for (size_t i = 1; i < arr.size(); i++) {
    if (arr[i] < arr[i - 1]) {
      std::swap(arr[i], arr[i - 1]);
    }
  }
}

void chernova_n_int_radix_sort_batcher_omp::TestTaskOpenMP::MergeSortBatcher(std::vector<int> &arr, size_t part_size) {
  if (arr.size() <= part_size) {
    RadixSort(arr);
    return;
  }

  const size_t mid = arr.size() / 2 + (arr.size() / 2) % 2;
  std::vector<int> left(arr.begin(), arr.begin() + mid);
  std::vector<int> right(arr.begin() + mid, arr.end());

#pragma omp task shared(left)
  {
    MergeSortBatcher(left, part_size);
  }

#pragma omp task shared(right)
  {
    MergeSortBatcher(right, part_size);
  }

#pragma omp taskwait

  std::copy(left.begin(), left.end(), arr.begin());
  std::copy(right.begin(), right.end(), arr.begin() + mid);

  std::vector<int> merge_buf(left.size());
  Merge(arr, merge_buf, left.size(), right.size(), 0);
  Merge(arr, merge_buf, left.size(), right.size(), 1);
  ComparePairs(arr);
}

bool chernova_n_int_radix_sort_batcher_omp::TestTaskOpenMP::PostProcessingImpl() {
  for (size_t i = 0; i < output_size; i++) {
    reinterpret_cast<int *>(task_data->outputs[0])[i] = input_[i];
  }
  return true;
}