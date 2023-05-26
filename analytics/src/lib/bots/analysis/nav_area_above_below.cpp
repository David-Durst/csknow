//
// Created by durst on 5/25/23.
//

#include "bots/analysis/nav_area_above_below.h"
#include <filesystem>

namespace csknow::nav_area_above_below {
    NavAreaAboveBelow::NavAreaAboveBelow(const MapMeshResult &mapMeshResult, const std::string &navPath) {
        std::string filePath = navPath + "/de_dust2_nav_above_below.hdf5";
        if (std::filesystem::exists(filePath)) {
            load(filePath);
        }
        else {
            computeNavRegion(mapMeshResult);
            size_t numXSteps = static_cast<size_t>(std::ceil((navRegion.max.x - navRegion.min.x) / step_size));
            size_t numYSteps = static_cast<size_t>(std::ceil((navRegion.max.y - navRegion.min.y) / step_size));
            size_t numZSteps = static_cast<size_t>(std::ceil((navRegion.max.z - navRegion.min.z) / step_size));
            for (size_t xStep = 0; xStep < numXSteps; xStep++) {
                for (size_t yStep = 0; yStep < numYSteps; yStep++) {
                    for (size_t zStep = 0; zStep < numZSteps; zStep++) {
                        Vec3 pos {
                            navRegion.min.x + (static_cast<double>(xStep) + 0.5) * step_size,
                            navRegion.min.y + (static_cast<double>(yStep) + 0.5) * step_size,
                            navRegion.min.z + (static_cast<double>(zStep) + 0.5) * step_size,
                        };
                        MapMeshResult::OverlappingResult overlappingResult = mapMeshResult.overlappingAreas(pos);
                        AreaId areaAboveId = 0;
                        float minZAbove = std::numeric_limits<float>::max() * -1.;
                        bool foundZAbove = false;
                        AreaId areaBelowId = 0;
                        float maxZBelow = std::numeric_limits<float>::max();
                        bool foundZBelow = false;
                        AreaId areaNearestId = 0;
                        float localZNearest = std::numeric_limits<float>::max();
                        bool foundZNearest = false;
                        // look for non-overlapping area above or below
                        for (const auto & overlappingAreaId : overlappingResult.overlappingIn2D) {
                            const AABB & area = mapMeshResult.coordinate[mapMeshResult.areaToInternalId.at(overlappingAreaId)];
                            if (pos.z < area.min.z && area.min.z < minZAbove) {
                                areaAboveId = overlappingAreaId;
                                minZAbove = area.min.z;
                                foundZAbove = true;
                            }
                            if (pos.z > area.max.z && area.max.z > maxZBelow) {
                                areaBelowId = overlappingAreaId;
                                maxZBelow = area.max.z;
                                foundZBelow = true;
                            }
                            // assume non-overlapping, will handle overlapping later
                            double zDistance = std::min(std::abs(pos.z - area.min.z), std::abs(pos.z - area.max.z));
                            if (zDistance < std::abs(pos.z - localZNearest)) {
                                areaNearestId = overlappingAreaId;
                                if (pos.z < area.min.z) {
                                    localZNearest = area.min.z;
                                }
                                else {
                                    localZNearest = area.max.z;
                                }
                                foundZNearest = true;
                            }
                        }
                        // if no nonoverlapping areas, take any overlapping areas
                        if (!foundZAbove && !overlappingResult.overlappingIn3D.empty()) {
                            areaAboveId = overlappingResult.overlappingIn3D.front();
                            minZAbove = pos.z;
                            foundZAbove = true;
                        }
                        if (!foundZBelow && !overlappingResult.overlappingIn3D.empty()) {
                            areaBelowId = overlappingResult.overlappingIn3D.front();
                            maxZBelow = pos.z;
                            foundZBelow = true;
                        }
                        // if overlapping, that's nearest, otherwise get nearest
                        if (!overlappingResult.overlappingIn3D.empty()) {
                            areaNearestId = overlappingResult.overlappingIn3D.front();
                            localZNearest = pos.z;
                            foundZNearest = true;
                        }
                        areaAbove.push_back(areaAboveId);
                        zAbove.push_back(minZAbove);
                        foundAbove.push_back(foundZAbove);
                        areaBelow.push_back(areaBelowId);
                        zBelow.push_back(maxZBelow);
                        foundBelow.push_back(foundZBelow);
                        areaNearest.push_back(areaNearestId);
                        zNearest.push_back(localZNearest);
                        foundNearest.push_back(foundZNearest);
                    }
                }
            }
            save(filePath);
        }
    }

    void NavAreaAboveBelow::save(const std::string &filePath) {
        // We create an empty HDF55 file, by truncating an existing
        // file if required:
        HighFive::File file(filePath, HighFive::File::Overwrite);
        HighFive::DataSetCreateProps hdf5CreateProps;
        hdf5CreateProps.add(HighFive::Deflate(3));
        hdf5CreateProps.add(HighFive::Chunking(areaBelow.size()));

        file.createDataSet("/data/area above", areaAbove, hdf5CreateProps);
        file.createDataSet("/data/z above", zAbove, hdf5CreateProps);
        file.createDataSet("/data/found above", foundAbove, hdf5CreateProps);
        file.createDataSet("/data/area below", areaBelow, hdf5CreateProps);
        file.createDataSet("/data/z below", zBelow, hdf5CreateProps);
        file.createDataSet("/data/found below", foundBelow, hdf5CreateProps);
        file.createDataSet("/data/area nearest", areaNearest, hdf5CreateProps);
        file.createDataSet("/data/z nearest", zNearest, hdf5CreateProps);
        file.createDataSet("/data/found nearest", foundBelow, hdf5CreateProps);

        vector<double> navRegionVec{navRegion.min.x, navRegion.min.y, navRegion.min.z,
                                    navRegion.max.x, navRegion.max.y, navRegion.max.z};
        file.createDataSet("/extra/nav region", navRegionVec);
    }

    void NavAreaAboveBelow::load(const std::string &filePath) {
        HighFive::File file(filePath, HighFive::File::ReadOnly);

        areaAbove = file.getDataSet("/data/area above").read<std::vector<AreaId>>();
        zAbove = file.getDataSet("/data/z above").read<std::vector<float>>();
        foundAbove = file.getDataSet("/data/found above").read<std::vector<bool>>();
        areaBelow = file.getDataSet("/data/area below").read<std::vector<AreaId>>();
        zBelow = file.getDataSet("/data/z below").read<std::vector<float>>();
        foundBelow = file.getDataSet("/data/found below").read<std::vector<bool>>();
        auto navRegionVec = file.getDataSet("/extra/nav region").read<std::vector<double>>();
        navRegion = AABB{{navRegionVec[0], navRegionVec[1], navRegionVec[2]},
                         {navRegionVec[3], navRegionVec[4], navRegionVec[5]}};
    }

    void NavAreaAboveBelow::computeNavRegion(const MapMeshResult &mapMeshResult) {
        navRegion = {
                {
                    std::numeric_limits<double>::max(),
                    std::numeric_limits<double>::max(),
                    std::numeric_limits<double>::max()
                },
                {
                    -1 * std::numeric_limits<double>::max(),
                    -1 * std::numeric_limits<double>::max(),
                    -1 * std::numeric_limits<double>::max()
                }
        };

        for (const auto & area : mapMeshResult.coordinate) {
            navRegion.min = min(navRegion.min, area.min);
            navRegion.max = max(navRegion.max, area.max);
        }
        navRegion.min.x -= WIDTH;
        navRegion.min.y -= WIDTH;
        navRegion.max.x += WIDTH;
        navRegion.max.y += WIDTH;
        navRegion.max.z += STANDING_HEIGHT;
        std::cout << "map range " << navRegion.toString() << std::endl;
    }

}