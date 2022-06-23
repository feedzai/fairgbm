/**
 * The copyright of this file belongs to Feedzai. The file cannot be
 * reproduced in whole or in part, stored in a retrieval system,
 * transmitted in any form, or by any means electronic, mechanical,
 * photocopying, or otherwise, without the prior permission of the owner.
 *
 * (c) 2021 Feedzai, Strictly Confidential
 */

#ifndef LIGHTGBM_CONSTRAINED_HPP
#define LIGHTGBM_CONSTRAINED_HPP

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

    inline double sigmoid(double x) {
      return 1. / (1. + std::exp(-x));
    }

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

    template<typename T, typename Allocator = std::allocator<T>>
    void write_values(const std::string& dir, const std::string& filename,
                      std::vector<T, Allocator> values) {
      struct stat buf;

      std::string filename_path = dir + "/" + filename;
      bool file_exists = (stat(filename_path.c_str(), &buf) != -1);

      std::stringstream tmp_buf;
      for (auto e : values) {
        tmp_buf << e << ",";
      }

      tmp_buf.seekp(-1, tmp_buf.cur);
      tmp_buf << std::endl;

      std::ofstream outfile;
      outfile.open(filename_path, std::ios::out | (file_exists ? std::ios::app : std::ios::trunc));
      outfile << tmp_buf.str();

      outfile.close();
    }

    template<typename T>
    void write_values(const std::string& dir, const std::string& filename, const T* arr, int arr_len) {
      struct stat buf;

      std::string filename_path = dir + "/" + filename;
      bool file_exists = (stat(filename_path.c_str(), &buf) != -1);

      std::stringstream tmp_buf;
      for (int i = 0; i < arr_len; i++) {
        tmp_buf << arr[i] << ",";
      }

      tmp_buf.seekp(-1, tmp_buf.cur);
      tmp_buf << std::endl;

      std::ofstream outfile;
      outfile.open(filename_path, std::ios::out | (file_exists ? std::ios::app : std::ios::trunc));
      outfile << tmp_buf.str();

      outfile.close();
    }

}

#endif //LIGHTGBM_CONSTRAINED_HPP
