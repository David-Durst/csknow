//
// Created by durst on 5/25/23.
//

#ifndef CSKNOW_NAV_AREA_ABOVE_BELOW_H
#define CSKNOW_NAV_AREA_ABOVE_BELOW_H

#include "geometryNavConversions.h"
#include "queries/reachable.h"

namespace csknow::nav_area_above_below {
    constexpr double step_size = 10.;
    class NavAreaAboveBelow {
    public:
        // below and above are strict, below or in is like gravity, falling unless standing inside
        vector<AreaId> areaAbove, areaBelow, areaBelowOrIn, areaNearest;
        vector<float> zAbove, zBelow, zBelowOrIn, zNearest;
        vector<bool> foundAbove, foundBelow, foundBelowOrIn, foundNearest;
        AABB navRegion;

        NavAreaAboveBelow(const MapMeshResult & mapMeshResult, const string& navPath);

        void save(const string& filePath);
        void load(const string& filePath);
        void computeNavRegion(const MapMeshResult & mapMeshResult);
        size_t posToIndex(Vec3 pos);
    };
}

#endif //CSKNOW_NAV_AREA_ABOVE_BELOW_H
