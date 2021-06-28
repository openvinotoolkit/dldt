// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifdef CPU_DEBUG_CAPS

#include "mkldnn_node.h"
#include "dnnl_debug.h"

namespace MKLDNNPlugin {
/**
 * Print node verbose execution information to cout.
 * Similiar to DNNL_VERBOSE output
 * Formating written in C using oneDNN format functions.
 * Can be rewritten in pure C++ if necessary
 */
void print(const MKLDNNNodePtr& node, const std::string& verboseLvl) {
    // use C stoi version to avoid dealing with exceptions
    if (verboseLvl.empty() || stoi(verboseLvl) < 1)
        return;

    if (node->isConstant() ||
        node->getType() == Input || node->getType() == Output)
        return;

    const int CPU_VERBOSE_DAT_LEN = 512;
    char portsInfo[CPU_VERBOSE_DAT_LEN] = {'\0'};
    int written = 0;
    int written_total = 0;

    auto shift = [&](int size) {
        if (written < 0 || written_total + size > CPU_VERBOSE_DAT_LEN) {
            const char* errorMsg = "# NOT ENOUGHT BUFFER SIZE #";
            snprintf(portsInfo, strlen(errorMsg), "%s", errorMsg);
            written_total = strlen(errorMsg);
            return;
        }

        written_total += size;
    };

    for (int i = 0; i < node->getParentEdges().size(); i++) {
        std::string prefix("src:" + std::to_string(i) + ':');
        const auto& srcMemDesc = node->getParentEdgeAt(i)->getMemory().GetDescriptor().data;
        written = snprintf(portsInfo + written_total, CPU_VERBOSE_DAT_LEN - written_total, " ");
        shift(written);
        written = snprintf(portsInfo + written_total, CPU_VERBOSE_DAT_LEN - written_total, "%s", prefix.c_str());
        shift(written);
        written = dnnl_md2fmt_str(portsInfo + written_total, CPU_VERBOSE_DAT_LEN - written_total, &srcMemDesc);
        shift(written);
        written = snprintf(portsInfo + written_total, CPU_VERBOSE_DAT_LEN - written_total, ":");
        shift(written);
        written = dnnl_md2dim_str(portsInfo + written_total, CPU_VERBOSE_DAT_LEN - written_total, &srcMemDesc);
        shift(written);
    }

    for (int i = 0; i < node->getChildEdges().size(); i++) {
        std::string prefix("dst:" + std::to_string(i) + ':');
        const auto& dstMemDesc = node->getChildEdgeAt(0)->getMemory().GetDescriptor().data;
        written = snprintf(portsInfo + written_total, CPU_VERBOSE_DAT_LEN - written_total, " ");
        shift(written);
        written = snprintf(portsInfo + written_total, CPU_VERBOSE_DAT_LEN - written_total, "%s", prefix.c_str());
        shift(written);
        written = dnnl_md2fmt_str(portsInfo + written_total, CPU_VERBOSE_DAT_LEN - written_total, &dstMemDesc);
        shift(written);
        written = snprintf(portsInfo + written_total, CPU_VERBOSE_DAT_LEN - written_total, ":");
        shift(written);
        written = dnnl_md2dim_str(portsInfo + written_total, CPU_VERBOSE_DAT_LEN - written_total, &dstMemDesc);
        shift(written);
    }

    std::string post_ops;
    if (!node->getFusedWith().empty()) {
        post_ops += "post_ops:'";
        for (const auto& fusedNode : node->getFusedWith()) {
            post_ops.append(fusedNode->getName()).append(":")
                .append(NameFromType(fusedNode->getType())).append(":")
                .append(algToString(fusedNode->getAlgorithm()))
                .append(";");
        }
        post_ops += "'";
    }

    std::cout << "ov_cpu_verbose,exec,cpu,"
              << node->getName() << ":" << NameFromType(node->getType()) << ','
              << impl_type_to_string(node->getSelectedPrimitiveDescriptor()->getImplementationType()) << ','
              << portsInfo << ','
              << algToString(node->getAlgorithm()) << ','
              << post_ops << ','
              << node->PerfCounter().duration().count()
              << "\n";
}
} // namespace MKLDNNPlugin
#endif // CPU_DEBUG_CAPS
