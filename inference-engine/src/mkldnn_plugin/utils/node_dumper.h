// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifdef CPU_DEBUG_CAPS
#pragma once

#include "mkldnn_node.h"

#include <array>
#include <string>

namespace MKLDNNPlugin {

/**
 * Blobs are not dumped by default
 * Blobs are dumped if node matches all specified env filters
 *
 * To dump blobs from all the nodes use the following filter:
 *
 * OV_CPU_BLOB_DUMP_NODE_NAME=.+
 */
class NodeDumper {
public:
    NodeDumper(int _count) : count(_count) {
        setup();
    }

    void setup();

    void dumpInputBlobs(const MKLDNNNodePtr &node) const;

    void dumpOutputBlobs(const MKLDNNNodePtr &node) const;

private:
    void dumpInternalBlobs(const MKLDNNNodePtr& node) const;
    bool shouldBeDumped(const MKLDNNNodePtr &node) const;

    bool shouldDumpAsText = false;
    bool shouldDumpInternalBlobs = false;
    int count;

    std::string dumpDirName = "mkldnn_dump";

    enum FILTER {
        BY_EXEC_ID,
        BY_TYPE,
        BY_LAYER_TYPE,
        BY_NAME,
        COUNT,
    };

    std::array<std::string, FILTER::COUNT> dumpFilters;
};
} // namespace MKLDNNPlugin
#endif // CPU_DEBUG_CAPS
