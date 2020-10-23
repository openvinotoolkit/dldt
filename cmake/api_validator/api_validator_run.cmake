# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_policy(SET CMP0012 NEW)

foreach(var UWP_API_VALIDATOR UWP_API_VALIDATOR_TARGET
            UWP_API_VALIDATOR_APIS UWP_API_VALIDATOR_EXCLUSION
            UWP_API_VALIDATOR_OUTPUT)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "Variable ${var} is not defined")
    endif()
endforeach()

# create command

set(command "${UWP_API_VALIDATOR}"
        -SupportedApiXmlFiles:${UWP_API_VALIDATOR_APIS}
        -DriverPackagePath:${UWP_API_VALIDATOR_TARGET})
if(EXISTS "${UWP_API_VALIDATOR_EXCLUSION}")
    list(APPEND command
        -BinaryExclusionListXmlFile:${UWP_API_VALIDATOR_EXCLUSION}
        -StrictCompliance:TRUE)
    set(UWP_HAS_BINARY_EXCLUSION ON)
endif()

# execute

execute_process(COMMAND ${command}
                OUTPUT_VARIABLE output_message
                ERROR_VARIABLE error_message
                RESULT_VARIABLE exit_code
                OUTPUT_STRIP_TRAILING_WHITESPACE)

file(WRITE "${UWP_API_VALIDATOR_OUTPUT}" "${output_message}\n\n\n${error_message}")

# post-process output

if(ON OR NOT UWP_HAS_BINARY_EXCLUSION)
    get_filename_component(name "${UWP_API_VALIDATOR_TARGET}" NAME)
    set(exclusion_dlls "msvcp140.dll" "vcruntime140.dll")

    # remove exclusions from error_message

    foreach(dll IN LISTS exclusion_dlls)
        string(REGEX REPLACE
                "ApiValidation: Error: ${name} has unsupported API call to \"${dll}![^\"]+\"\n"
                "" error_message "${error_message}")
    endforeach()

    # throw error if error_message still contains any errors

    if(error_message)
        message("${error_message}")
    endif()
endif()

# write output

if(UWP_HAS_BINARY_EXCLUSION AND NOT exit_code EQUAL 0)
    message("${error_message}")
endif()
