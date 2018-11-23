// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <memory>
#include <map>
#include <string>
#include <vector>
#include <utility>

#include <inference_engine.hpp>
#include <format_reader_ptr.h>

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>

#include "benchmark_app.h"

using namespace InferenceEngine;

long long getDurationInNanoseconds(const std::string& device);

double getMedianValue(const std::vector<float>& sortedTimes);

void fillBlobWithImage(
    Blob::Ptr& inputBlob,
    const std::vector<std::string>& filePaths,
    const size_t batchSize,
    const InferenceEngine::InputInfo& info);

static const std::vector<std::pair<std::string, long long>> deviceDurationsInSeconds{
    { "CPU", 60LL },
    { "GPU", 60LL },
    { "VPU", 60LL },
    { "MYRIAD", 60LL },
    { "FPGA", 120LL },
    { "UNKNOWN", 120LL }
};

/**
* @brief The entry point the benchmark application
*/
int main(int argc, char *argv[]) {
    try {
        slog::info << "InferenceEngine: " << InferenceEngine::GetInferenceEngineVersion() << slog::endl;

        slog::info << "Parsing input parameters" << slog::endl;
        gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
        if (FLAGS_h) {
            showUsage();
            return 0;
        }

        if (FLAGS_m.empty()) {
            throw std::logic_error("Model required is not set. Please use -h.");
        }

        if (FLAGS_api.empty()) {
            throw std::logic_error("API not selected. Please use -h.");
        }

        if (FLAGS_api != "async" && FLAGS_api != "sync") {
            throw std::logic_error("Incorrect API. Please use -h.");
        }

        if (FLAGS_i.empty()) {
            throw std::logic_error("Input is not set. Please use -h.");
        }

        if (FLAGS_niter < 0) {
            throw std::logic_error("Number of iterations should be positive (invalid -niter option value)");
        }

        if (FLAGS_nireq < 0) {
            throw std::logic_error("Number of inference requests should be positive (invalid -nireq option value)");
        }

        if (FLAGS_b < 0) {
            throw std::logic_error("Batch size should be positive (invalid -b option value)");
        }

        std::vector<std::string> inputs;
        parseInputFilesArguments(inputs);
        if (inputs.size() == 0ULL) {
            throw std::logic_error("no images found");
        }

        // --------------------------- 1. Load Plugin for inference engine -------------------------------------

        slog::info << "Loading plugin" << slog::endl;
        InferencePlugin plugin = PluginDispatcher({ FLAGS_pp }).getPluginByDevice(FLAGS_d);

        if (!FLAGS_l.empty()) {
            // CPU (MKLDNN) extensions is loaded as a shared library and passed as a pointer to base extension
            const std::shared_ptr<IExtension> extension_ptr = InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(FLAGS_l);
            plugin.AddExtension(extension_ptr);
            slog::info << "CPU (MKLDNN) extensions is loaded " << FLAGS_l << slog::endl;
        } else if (!FLAGS_c.empty()) {
            // Load clDNN Extensions
            plugin.SetConfig({ {CONFIG_KEY(CONFIG_FILE), FLAGS_c} });
            slog::info << "GPU extensions is loaded " << FLAGS_c << slog::endl;
        }

        InferenceEngine::ResponseDesc resp;

        const Version *pluginVersion = plugin.GetVersion();
        slog::info << pluginVersion << slog::endl << slog::endl;

        // --------------------------- 2. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------

        slog::info << "Loading network files" << slog::endl;

        InferenceEngine::CNNNetReader netBuilder;
        netBuilder.ReadNetwork(FLAGS_m);
        const std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
        netBuilder.ReadWeights(binFileName);

        InferenceEngine::CNNNetwork cnnNetwork = netBuilder.getNetwork();
        const InferenceEngine::InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
        if (inputInfo.empty()) {
            throw std::logic_error("no inputs info is provided");
        }

        if (inputInfo.size() != 1) {
            throw std::logic_error("only one input layer network is supported");
        }

        // --------------------------- 3. Resize network to match image sizes and given batch----------------------
        if (FLAGS_b != 0) {
            // We support models having only one input layers
            ICNNNetwork::InputShapes shapes = cnnNetwork.getInputShapes();
            const ICNNNetwork::InputShapes::iterator& it = shapes.begin();
            if (it->second.size() != 4) {
                throw std::logic_error("Unsupported model for batch size changing in automatic mode");
            }
            it->second[0] = FLAGS_b;
            slog::info << "Resizing network to batch = " << FLAGS_b << slog::endl;
            cnnNetwork.reshape(shapes);
        }

        const size_t batchSize = cnnNetwork.getBatchSize();
        const Precision precision = inputInfo.begin()->second->getPrecision();
        slog::info << (FLAGS_b != 0 ? "Network batch size was changed to: " : "Network batch size: ") << batchSize <<
            ", precision: " << precision << slog::endl;

        // --------------------------- 4. Configure input & output ---------------------------------------------

        const InferenceEngine::Precision inputPrecision = InferenceEngine::Precision::U8;
        for (auto& item : inputInfo) {
            /** Set the precision of input data provided by the user, should be called before load of the network to the plugin **/
            item.second->setInputPrecision(inputPrecision);
        }

        const size_t imagesCount = inputs.size();
        if (batchSize > imagesCount) {
            slog::warn << "Network batch size " << batchSize << " is greater than images count " << imagesCount <<
                ", some input files will be duplicated" << slog::endl;
        } else if (batchSize < imagesCount) {
            slog::warn << "Network batch size " << batchSize << " is less then images count " << imagesCount <<
                ", some input files will be ignored" << slog::endl;
        }

        // ------------------------------ Prepare output blobs -------------------------------------------------
        slog::info << "Preparing output blobs" << slog::endl;
        InferenceEngine::OutputsDataMap outputInfo(cnnNetwork.getOutputsInfo());
        InferenceEngine::BlobMap outputBlobs;
        for (auto& item : outputInfo) {
            const InferenceEngine::DataPtr outData = item.second;
            if (!outData) {
                throw std::logic_error("output data pointer is not valid");
            }
            InferenceEngine::SizeVector outputDims = outData->dims;
            const InferenceEngine::Precision outputPrecision = InferenceEngine::Precision::FP32;

            /** Set the precision of output data provided by the user, should be called before load of the network to the plugin **/
            outData->precision = outputPrecision;
            InferenceEngine::TBlob<float>::Ptr output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;
        }

        // --------------------------- 5. Loading model to the plugin ------------------------------------------

        slog::info << "Loading model to the plugin" << slog::endl;
        const std::map<std::string, std::string> networkConfig;
        InferenceEngine::ExecutableNetwork exeNetwork = plugin.LoadNetwork(cnnNetwork, networkConfig);

        // --------------------------- 6. Performance measurements stuff ------------------------------------------

        typedef std::chrono::high_resolution_clock Time;
        typedef std::chrono::nanoseconds ns;

        std::vector<float> times;
        long long durationInNanoseconds;
        if (FLAGS_niter != 0) {
            durationInNanoseconds = 0LL;
            times.reserve(FLAGS_niter);
        } else {
            durationInNanoseconds = getDurationInNanoseconds(FLAGS_d);
        }

        if (FLAGS_api == "sync") {
            InferRequest inferRequest = exeNetwork.CreateInferRequest();
            slog::info << "Sync request created" << slog::endl;

            for (const InputsDataMap::value_type& item : inputInfo) {
                Blob::Ptr inputBlob = inferRequest.GetBlob(item.first);
                fillBlobWithImage(inputBlob, inputs, batchSize, *item.second);
            }

            if (FLAGS_niter != 0) {
                slog::info << "Start inference synchronously (" << FLAGS_niter << " sync inference executions)" << slog::endl << slog::endl;
            } else {
                slog::info << "Start inference synchronously (" << durationInNanoseconds * 0.000001 << " ms duration)" << slog::endl << slog::endl;
            }

            const auto startTime = Time::now();
            auto currentTime = Time::now();

            size_t iteration = 0ULL;
            while ((iteration < FLAGS_niter) || ((FLAGS_niter == 0LL) && ((currentTime - startTime).count() < durationInNanoseconds))) {
                const auto iterationStartTime = Time::now();
                inferRequest.Infer();
                currentTime = Time::now();

                const auto iterationDurationNs = std::chrono::duration_cast<ns>(currentTime - iterationStartTime);
                times.push_back(static_cast<double>(iterationDurationNs.count()) * 0.000001);

                iteration++;
            }

            std::sort(times.begin(), times.end());
            const double latency = getMedianValue(times);
            slog::info << "Latency: " << latency << " ms" << slog::endl;

            slog::info << "Throughput: " << batchSize * 1000.0 / latency << " FPS" << slog::endl;
        } else if (FLAGS_api == "async") {
            std::vector<InferRequest> inferRequests;
            inferRequests.reserve(FLAGS_nireq);

            for (size_t i = 0; i < FLAGS_nireq; i++) {
                InferRequest inferRequest = exeNetwork.CreateInferRequest();
                inferRequests.push_back(inferRequest);

                for (const InputsDataMap::value_type& item : inputInfo) {
                    Blob::Ptr inputBlob = inferRequest.GetBlob(item.first);
                    fillBlobWithImage(inputBlob, inputs, batchSize, *item.second);
                }
            }

            if (FLAGS_niter != 0) {
                slog::info << "Start inference asynchronously (" << FLAGS_niter <<
                    " async inference executions, " << FLAGS_nireq <<
                    " inference requests in parallel)" << slog::endl << slog::endl;
            } else {
                slog::info << "Start inference asynchronously (" << durationInNanoseconds * 0.000001 <<
                    " ms duration, " << FLAGS_nireq <<
                    " inference requests in parallel)" << slog::endl << slog::endl;
            }

            size_t currentInference = 0ULL;
            bool requiredInferenceRequestsWereExecuted = false;
            long long previousInference = 1LL - FLAGS_nireq;

            // warming up - out of scope
            inferRequests[0].StartAsync();
            inferRequests[0].Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);

            const size_t stepsCount = FLAGS_niter + FLAGS_nireq - 1;

            /** Start inference & calculate performance **/
            const auto startTime = Time::now();

            size_t step = 0ULL;
            while ((!requiredInferenceRequestsWereExecuted) ||
                (step < stepsCount) ||
                ((FLAGS_niter == 0LL) && ((Time::now() - startTime).count() < durationInNanoseconds))) {
                // start new inference
                inferRequests[currentInference].StartAsync();

                // wait the latest inference execution if exists
                if (previousInference >= 0) {
                    const StatusCode code = inferRequests[previousInference].Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
                    if (code != StatusCode::OK) {
                        throw std::logic_error("Wait");
                    }
                }

                currentInference++;
                if (currentInference >= FLAGS_nireq) {
                    currentInference = 0;
                    requiredInferenceRequestsWereExecuted = true;
                }

                previousInference++;
                if (previousInference >= FLAGS_nireq) {
                    previousInference = 0;
                }

                step++;
            }

            // wait the latest inference executions
            for (size_t notCompletedIndex = 0ULL; notCompletedIndex < (FLAGS_nireq - 1); ++notCompletedIndex) {
                if (previousInference >= 0) {
                    const StatusCode code = inferRequests[previousInference].Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
                    if (code != StatusCode::OK) {
                        throw std::logic_error("Wait");
                    }
                }

                previousInference++;
                if (previousInference >= FLAGS_nireq) {
                    previousInference = 0LL;
                }
            }

            const double totalDuration = std::chrono::duration_cast<ns>(Time::now() - startTime).count() * 0.000001;
            const double fps = batchSize * 1000.0 * step / totalDuration;
            slog::info << "Throughput: " << fps << " FPS" << slog::endl;
        } else {
            throw std::logic_error("unknown api command line argument value");
        }
    } catch (const std::exception& ex) {
        slog::err << ex.what() << slog::endl;
        return 3;
    }

    return 0;
}

long long getDurationInNanoseconds(const std::string& device) {
    auto duration = 0LL;
    for (const auto& deviceDurationInSeconds : deviceDurationsInSeconds) {
        if (device.find(deviceDurationInSeconds.first) != std::string::npos) {
            duration = std::max(duration, deviceDurationInSeconds.second);
        }
    }

    if (duration == 0LL) {
        const auto unknownDeviceIt = find_if(
            deviceDurationsInSeconds.begin(),
            deviceDurationsInSeconds.end(),
            [](std::pair<std::string, long long> deviceDuration) { return deviceDuration.first == "UNKNOWN"; });

        if (unknownDeviceIt == deviceDurationsInSeconds.end()) {
            throw std::logic_error("UNKNOWN device was not found in device duration list");
        }
        duration = unknownDeviceIt->second;
        slog::warn << "Default duration " << duration << " seconds for unknown device '" << device << "' is used" << slog::endl;
    }

    return duration * 1000000000LL;
}

double getMedianValue(const std::vector<float>& sortedTimes) {
    return (sortedTimes.size() % 2 != 0) ?
        sortedTimes[sortedTimes.size() / 2ULL] :
        (sortedTimes[sortedTimes.size() / 2ULL] + sortedTimes[sortedTimes.size() / 2ULL - 1ULL]) / 2.0;
}

void fillBlobWithImage(
    Blob::Ptr& inputBlob,
    const std::vector<std::string>& filePaths,
    const size_t batchSize,
    const InferenceEngine::InputInfo& info) {

    uint8_t* inputBlobData = inputBlob->buffer().as<uint8_t*>();
    const SizeVector& inputBlobDims = inputBlob->dims();

    slog::info << "Input dimensions (" << info.getTensorDesc().getLayout() << "): ";
    for (const auto& i : info.getTensorDesc().getDims()) {
        slog::info << i << " ";
    }
    slog::info << slog::endl;

    /** Collect images data ptrs **/
    std::vector<std::shared_ptr<uint8_t>> vreader;
    vreader.reserve(batchSize);

    for (size_t i = 0ULL, inputIndex = 0ULL; i < batchSize; i++, inputIndex++) {
        if (inputIndex >= filePaths.size()) {
            inputIndex = 0ULL;
        }

        FormatReader::ReaderPtr reader(filePaths[inputIndex].c_str());
        if (reader.get() == nullptr) {
            slog::warn << "Image " << filePaths[inputIndex] << " cannot be read!" << slog::endl << slog::endl;
            continue;
        }

        /** Getting image data **/
        std::shared_ptr<uint8_t> imageData(reader->getData(info.getDims()[0], info.getDims()[1]));
        if (imageData) {
            vreader.push_back(imageData);
        }
    }

    /** Fill input tensor with images. First b channel, then g and r channels **/
    const size_t numChannels = inputBlobDims[2];
    const size_t imageSize = inputBlobDims[1] * inputBlobDims[0];
    /** Iterate over all input images **/
    for (size_t imageId = 0; imageId < vreader.size(); ++imageId) {
        /** Iterate over all pixel in image (b,g,r) **/
        for (size_t pid = 0; pid < imageSize; pid++) {
            /** Iterate over all channels **/
            for (size_t ch = 0; ch < numChannels; ++ch) {
                /**          [images stride + channels stride + pixel id ] all in bytes            **/
                inputBlobData[imageId * imageSize * numChannels + ch * imageSize + pid] = vreader.at(imageId).get()[pid*numChannels + ch];
            }
        }
    }
}
