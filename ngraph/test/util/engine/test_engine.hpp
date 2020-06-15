#pragma once

#include <ie_core.hpp>
#include "../../util/all_close.hpp"
#include "../../util/all_close_f.hpp"
#include "ngraph/function.hpp"

// Builds a class name for a given backend prefix
// The prefix should come from cmake
// Example: INTERPRETER -> INTERPRETER_Engine
// Example: IE_CPU -> IE_CPU_Engine
#define ENGINE_CLASS_NAME(backend) backend##_Engine
namespace ngraph
{
    namespace test
    {
        // TODO - implement when IE_CPU engine is done
        class INTERPRETER_Engine
        {
        public:
            INTERPRETER_Engine(std::shared_ptr<Function> function) {}
            void infer() {}
            testing::AssertionResult compare_results(const size_t tolerance_bits)
            {
                return testing::AssertionSuccess();
            }
            template <typename T>
            void add_input(const Shape& shape, const std::vector<T>& values)
            {
            }
            template <typename T>
            void add_expected_output(ngraph::Shape expected_shape, const std::vector<T>& values)
            {
            }
        };

        // TODO -inherit from IE_CPU_Engine?
        using IE_GPU_Engine = INTERPRETER_Engine;

        class IE_CPU_Engine
        {
        public:
            IE_CPU_Engine() = delete;
            IE_CPU_Engine(std::shared_ptr<Function> function);

            void infer()
            {
                if (m_network_inputs.size() != m_allocated_inputs)
                {
                    THROW_IE_EXCEPTION << "The tested graph has " << m_network_inputs.size()
                                       << " inputs, but " << m_allocated_inputs << " were passed.";
                }
                else
                {
                    m_inference_req.Infer();
                }
            };

            testing::AssertionResult
                compare_results(const size_t tolerance_bits = DEFAULT_FLOAT_TOLERANCE_BITS)
            {
                auto comparison_result = testing::AssertionSuccess();

                for (const auto output : m_network_outputs)
                {
                    InferenceEngine::MemoryBlob::CPtr computed_output_blob =
                        InferenceEngine::as<InferenceEngine::MemoryBlob>(
                            m_inference_req.GetBlob(output.first));

                    const auto& expected_output_blob = m_expected_outputs[output.first];

                    // TODO: assert that both blobs have the same precision?
                    const auto& precision = computed_output_blob->getTensorDesc().getPrecision();

                    // TODO: assert that both blobs have the same number of elements?
                    comparison_result = compare_blobs(
                        computed_output_blob, expected_output_blob, precision, tolerance_bits);

                    if (comparison_result == testing::AssertionFailure())
                    {
                        break;
                    }
                }

                return comparison_result;
            }

            template <typename T>
            void add_input(const Shape& shape, const std::vector<T>& values)
            {
                // Retrieve the next function parameter which has not been set yet.
                // The params are stored in a vector in the order of their creation.
                const auto& function_params = m_function->get_parameters();
                const auto& input_to_allocate = function_params[m_allocated_inputs];
                // TODO: check if input exists
                // Retrieve the corresponding CNNNetwork input using param's friendly name.
                // Here the inputs are stored in the map and are accessible by a string key.
                const auto& input_info = m_network_inputs[input_to_allocate->get_friendly_name()];

                auto blob =
                    std::make_shared<InferenceEngine::TBlob<T>>(input_info->getTensorDesc());
                blob->allocate();
                auto* blob_buffer = blob->wmap().template as<T*>();
                // TODO: assert blob->size() == values.size() ?
                std::copy(values.begin(), values.end(), blob_buffer);

                m_inference_req.SetBlob(input_to_allocate->get_friendly_name(), blob);

                ++m_allocated_inputs;
            }

            template <typename T>
            void add_expected_output(ngraph::Shape expected_shape, const std::vector<T>& values)
            {
                const auto& function_output =
                    m_function->get_results()[m_allocated_expected_outputs];
                // TODO: assert that function_output->get_friendly_name() is in network outputs
                const auto output_info = m_network_outputs[function_output->get_friendly_name()];
                auto blob =
                    std::make_shared<InferenceEngine::TBlob<T>>(output_info->getTensorDesc());
                blob->allocate();
                auto* blob_buffer = blob->wmap().template as<T*>();
                // TODO: assert blob->size() == values.size() ?
                std::copy(values.begin(), values.end(), blob_buffer);

                m_expected_outputs.emplace(function_output->get_friendly_name(), blob);

                ++m_allocated_expected_outputs;
            }

        private:
            std::shared_ptr<Function> m_function;
            InferenceEngine::InputsDataMap m_network_inputs;
            InferenceEngine::OutputsDataMap m_network_outputs;
            InferenceEngine::InferRequest m_inference_req;
            std::map<std::string, InferenceEngine::MemoryBlob::Ptr> m_expected_outputs;
            std::string m_output_name;
            unsigned int m_allocated_inputs = 0;
            unsigned int m_allocated_expected_outputs = 0;

            std::shared_ptr<Function>
                upgrade_and_validate_function(std::shared_ptr<Function> function) const;

            std::set<NodeTypeInfo> get_ie_ops() const;

            testing::AssertionResult compare_blobs(InferenceEngine::MemoryBlob::CPtr computed,
                                             InferenceEngine::MemoryBlob::CPtr expected,
                                             const InferenceEngine::Precision& precision,
                                             const size_t tolerance_bits) const
            {
                switch (static_cast<InferenceEngine::Precision::ePrecision>(precision))
                {
                case InferenceEngine::Precision::FP32:
                    return compare_blobs<float>(computed, expected, tolerance_bits);
                    break;
                case InferenceEngine::Precision::I8:
                    return compare_blobs<int8_t>(computed, expected, tolerance_bits);
                    break;
                case InferenceEngine::Precision::I16:
                    return compare_blobs<int8_t>(computed, expected, tolerance_bits);
                    break;
                case InferenceEngine::Precision::I32:
                    return compare_blobs<int16_t>(computed, expected, tolerance_bits);
                    break;
                case InferenceEngine::Precision::I64:
                    return compare_blobs<int64_t>(computed, expected, tolerance_bits);
                    break;
                case InferenceEngine::Precision::U8:
                    return compare_blobs<uint8_t>(computed, expected, tolerance_bits);
                    break;
                case InferenceEngine::Precision::U16:
                    return compare_blobs<uint16_t>(computed, expected, tolerance_bits);
                    break;
                case InferenceEngine::Precision::U64:
                    return compare_blobs<uint64_t>(computed, expected, tolerance_bits);
                    break;
                case InferenceEngine::Precision::BOOL:
                    return compare_blobs<uint8_t>(computed, expected, tolerance_bits);
                    break;
                default: THROW_IE_EXCEPTION << "Not implemented yet";
                }
            }

            template <typename T>
            typename std::enable_if<std::is_floating_point<T>::value,
                                    testing::AssertionResult>::type
                compare_blobs(InferenceEngine::MemoryBlob::CPtr computed,
                              InferenceEngine::MemoryBlob::CPtr expected,
                              const size_t tolerance_bits) const
            {
                const auto test_results = extract_test_results<T>(computed, expected);

                return ngraph::test::all_close_f(
                    test_results.first, test_results.second, tolerance_bits);
            }

            template <typename T>
            typename std::enable_if<std::is_integral<T>::value, testing::AssertionResult>::type
                compare_blobs(InferenceEngine::MemoryBlob::CPtr computed,
                              InferenceEngine::MemoryBlob::CPtr expected,
                              const size_t tolerance_bits) const
            {
                const auto test_results = extract_test_results<T>(computed, expected);

                return ngraph::test::all_close<T>(
                    test_results.first, test_results.second, tolerance_bits);
            }

            template <typename T>
            std::pair<std::vector<T>, std::vector<T>>
                extract_test_results(InferenceEngine::MemoryBlob::CPtr computed,
                                     InferenceEngine::MemoryBlob::CPtr expected) const
            {
                const auto computed_data = computed->rmap();
                const auto expected_data = expected->rmap();

                const auto* computed_data_buffer = computed_data.template as<const T*>();
                const auto* expected_data_buffer = computed_data.template as<const T*>();

                std::vector<T> computed_values(computed_data_buffer,
                                               computed_data_buffer + computed->size());
                std::vector<T> expected_values(expected_data_buffer,
                                               expected_data_buffer + computed->size());

                return std::make_pair(std::move(computed_values), std::move(expected_values));
            }
        };
    }
}