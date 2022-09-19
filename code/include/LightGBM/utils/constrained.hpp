/**
 * Copyright 2022 Feedzai
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#ifndef LIGHTGBM_UTILS_CONSTRAINED_HPP_
#define LIGHTGBM_UTILS_CONSTRAINED_HPP_

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <chrono>
#include <ctime>
#include <sstream>
#include <fstream>
#include <sys/stat.h>

namespace LightGBM {
namespace Constrained {

/**
 * Standard sigmoid mathematical function.
 * @param x the input to the function.
 * @return the sigmoid of the input.
 */
inline double sigmoid(double x) {
  return 1. / (1. + std::exp(-x));
}

/**
 * Finds the (key, value) pair with highest value.
 * @tparam Key The type of the Map Key.
 * @tparam Value The type of the Map Value.
 * @param x Reference to the map to search over.
 * @return The <K, V> pair with highest value V.
 */
template <class Key, class Value>
std::pair<Key, Value> findMaxValuePair(std::unordered_map<Key, Value> const &x)
{
  return *std::max_element(
          x.begin(), x.end(),
          [](const std::pair<Key, Value> &p1, const std::pair<Key, Value> &p2) {
              return p1.second < p2.second;
          }
  );
}

/**
 * Writes the given values to the end of the given file.
 * @tparam T The type of values in the input vector.
 * @tparam Allocator The type of allocator in the input vector.
 * @param dir The directory of the file to write on.
 * @param filename The name of the file to write on.
 * @param values A vector of the values to append to the file.
 */
template<typename T, typename Allocator = std::allocator<T>>
void write_values(const std::string& dir, const std::string& filename,
                  std::vector<T, Allocator> values) {
  struct stat buf;

  std::string filename_path = dir + "/" + filename;
  bool file_exists = (stat(filename_path.c_str(), &buf) != -1);

  std::ofstream outfile;
  outfile.open(filename_path, std::ios::out | (file_exists ? std::ios::app : std::ios::trunc));
  outfile << LightGBM::Common::Join(values, ",") << std::endl;

  outfile.close();
};
}
}

#endif  // LIGHTGBM_UTILS_CONSTRAINED_HPP_
