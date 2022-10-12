//
// Created by durst on 10/12/22.
//

#ifndef CSKNOW_OCTREE_H
#define CSKNOW_OCTREE_H

#include <vector>
#include <array>
#include <cstdint>
#include <optional>
#include "geometry.h"
using std::vector;
using std::array;
using std::optional;

typedef std::size_t OctreeIndex;

namespace csknow {
    class Octree {
        struct Node {
            std::array<optional<OctreeIndex>, 8> children;
            AABB extent;

            explicit
            Node(AABB extent) : extent(extent) {
                children.fill({});
            }
        };
        vector<Node> data;

    public:
        explicit
        Octree(AABB extent) : data{Node(extent)} { }

    };

}

#endif //CSKNOW_OCTREE_H
