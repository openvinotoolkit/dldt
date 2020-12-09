// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <fstream>
#include <regex>

#include "inference_engine.hpp"

#include "common_test_utils/file_utils.hpp"

#include "ops_cache.hpp"
#include "matchers/matchers_manager.hpp"
#include "gflag_config.hpp"

// TODO: Poor exceptions handling
int main(int argc, char *argv[]) {
    uint8_t ret_code = 0;

    try {
        gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
        if (FLAGS_h) {
            showUsage();
            return 0;
        }
// TODO: c++17 code
//    auto searcher = SubgraphsDumper::FolderIterator(FLAGS_input_folders, ".*.xml");
        auto ie = InferenceEngine::Core();

        auto cache = SubgraphsDumper::OPCache::make_cache();

        if (!CommonTestUtils::directoryExists(FLAGS_input_folders)) {
            std::string msg = "Input directory (" + FLAGS_input_folders + ") is not exist!";
            throw std::runtime_error(msg);
        }

        if (!CommonTestUtils::directoryExists(FLAGS_output_folders)) {
            std::string msg = "Output directory (" + FLAGS_output_folders + ") is not exist!";
            throw std::runtime_error(msg);
        }

        std::vector<std::string> input_folder_content;
        CommonTestUtils::directoryFileListRecursive(FLAGS_input_folders, input_folder_content);

// TODO: c++17 code
//    for (const auto &f : searcher.get_folder_content()) {
        for (const auto &file : input_folder_content) {
            if (CommonTestUtils::fileExists(file) && std::regex_match(file, std::regex(R"(.*\.xml)"))) {
                std::cout << "Processing model: " << file << std::endl;

                InferenceEngine::CNNNetwork net = ie.ReadNetwork(file);
                auto function = net.getFunction();
                cache->update_ops_cache(function, file);
            }
        }

        cache->serialize_cached_ops(FLAGS_output_folders);
    } catch (std::exception &e) {
        std::cerr << "Processing failed with exception:" << std::endl << e.what() << std::endl;
        ret_code = 2;
    }
    return ret_code;
}
