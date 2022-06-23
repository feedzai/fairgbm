/**
 * The copyright of this file belongs to Feedzai. The file cannot be
 * reproduced in whole or in part, stored in a retrieval system,
 * transmitted in any form, or by any means electronic, mechanical,
 * photocopying, or otherwise, without the prior permission of the owner.
 *
 * (c) 2021 Feedzai, Strictly Confidential
 */
/**
 * File intended to run an experiment that trains multiple LGBMs from different config files.
 */
#include <LightGBM/application.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <sys/stat.h>
#include <vector>


double time_it_with_chrono(std::string config_path, int argc) {
  std::cout << "-----------------------------------------------------------" << std::endl;
  std::cout << "--> Processing " << config_path << " file" << std::endl;
  std::cout << "-----------------------------------------------------------" << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  std::string prefixed_config_path = "config=" + config_path;
  char* argv[] = { (char*) "", (char*) prefixed_config_path.c_str() };
  LightGBM::Application app(argc, argv);
  app.Run();

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  return time_span.count();
}


void write_elapsed_time(std::string path, std::vector<std::string> paths, std::vector<double> times) {
  struct stat buf;
  bool file_exists = (stat(path.c_str(), &buf) != -1);

  std::stringstream tmp_buf;
  for (ushort i = 0; i < paths.size(); i++) {
    tmp_buf << paths[i] << "," << times[i] << std::endl;
  }
  std::ofstream outfile;
  outfile.open(path, std::ios::out | (file_exists ? std::ios::app : std::ios::trunc));
  outfile << tmp_buf.str();
  outfile.close();
}


int main() {
  bool success = false;
  try {

//    std::string experiments_root_path = "/home/andre.cruz/Documents/fair-boosting/experiments/";    // Path for local machine
    std::string experiments_root_path = "/mnt/home/andre.cruz/fair-boosting/experiments/";    // Path for hd-processor

//    std::string dataset_name = "Adult-2021";
//    std::string dataset_name = "AOF-Fairbench";
//    std::string dataset_name = "AOF-FairHO";
    std::string dataset_name = "AOF-FairHO-type_of_employment";

//    std::string experiment_name = "randomly-generated-configs";   // standard: run all configs

    /** AOF specific configurations **/
    std::string experiment_name = "randomly-generated-configs/LightGBM";
//    std::string experiment_name = "randomly-generated-configs/FairGBM";
//    std::string experiment_name = "randomly-generated-configs/LightGBM-with-unawareness";
//    std::string experiment_name = "randomly-generated-configs/LightGBM-with-equalized-prev";
//    std::string experiment_name = "randomly-generated-configs/FairGBM.BCE+BCE";
//    std::string experiment_name = "randomly-generated-configs/FairGBM.Recall+BCE";
//    std::string experiment_name = "randomly-generated-configs/FairGBM-with-equalized-prev.BCE+BCE";
//    std::string experiment_name = "randomly-generated-configs/FairGBM-with-unawareness.BCE+BCE";

    /** Adult specific configurations **/
//    std::string experiment_name = "randomly-generated-configs/LightGBM";
//    std::string experiment_name = "randomly-generated-configs/LightGBM-with-unawareness";
//    std::string experiment_name = "randomly-generated-configs/LightGBM-with-equalized-prev";
//    std::string experiment_name = "randomly-generated-configs/FairGBM-params-fixed";
//    std::string experiment_name = "randomly-generated-configs/FairGBM-params-exploration";
//    std::string experiment_name = "randomly-generated-configs/FairGBM-equalized-prev-params-exploration";

    std::string confs_root_path = experiments_root_path + dataset_name + "/confs/" + experiment_name + "/";
    std::string results_root_path = experiments_root_path + dataset_name + "/results/" + experiment_name + "/";

    std::vector<std::string> paths;
    int N_CONFIGS = 100;
    // Gather all config files under the given root folder
    for (int i = 0; i < N_CONFIGS; ++i) {
      std::stringstream ss;
      ss << std::setw(3) << std::setfill('0') << i << ".conf";
      std::string conf_file_path = confs_root_path + ss.str();
      paths.push_back(conf_file_path);
    }

    std::vector<double> elapsed_times;
    for (std::string path : paths) {
      double elapsed_time = time_it_with_chrono(path, 2);
      elapsed_times.push_back(elapsed_time);
    }

    write_elapsed_time(results_root_path + "elapsed-times.csv", paths, elapsed_times);

#ifdef USE_MPI
    LightGBM::Linkers::MpiFinalizeIfIsParallel();
#endif

    success = true;
  }
  catch (const std::exception &ex) {
    std::cerr << "Met Exceptions:" << std::endl;
    std::cerr << ex.what() << std::endl;
  }
  catch (const std::string &ex) {
    std::cerr << "Met Exceptions:" << std::endl;
    std::cerr << ex << std::endl;
  }
  catch (...) {
    std::cerr << "Unknown Exceptions" << std::endl;
  }

  if (!success) {
#ifdef USE_MPI
    LightGBM::Linkers::MpiAbortIfIsParallel();
#endif

    exit(-1);
  }
}
