//
// Created by durst on 5/25/23.
//

#include "bots/analysis/nav_area_above_below.h"
#include <filesystem>

namespace csknow::nav_area_above_below {
    NavAreaAboveBelow::NavAreaAboveBelow(const MapMeshResult &mapMeshResult, const std::string &navPath) {
        std::string filePath = navPath + "/d2_dust_nav_above_below.hdf5";
        if (std::filesystem::exists(filePath)) {
            load(mapMeshResult, filePath);
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
                        float minZAbove = std::numeric_limits<float>::max() * -1.;
                        bool foundZAbove = false;
                        float maxZBelow = std::numeric_limits<float>::max();
                        bool foundZBelow = false;
                        // look for non-overlapping area above or below
                        for (const auto & overlappingAreaId : overlappingResult.overlappingIn2D) {
                            const AABB & area = mapMeshResult.coordinate[mapMeshResult.areaToInternalId.at(overlappingAreaId)];
                            if (pos.z < area.min.z && (area.min.z < minZAbove)) {
                                minZAbove = area.min.z;
                                foundZAbove = true;
                            }
                            if (pos.z > area.max.z && (area.max.z > maxZBelow)) {
                                maxZBelow = area.max.z;
                                foundZBelow = true;
                            }
                        }
                        // if no nonoverlapping areas, take any overlapping areas
                        if (!foundZAbove && !overlappingResult.overlappingIn3D.empty()) {
                            minZAbove = pos.z;
                        }
                        if (!foundZBelow && !overlappingResult.overlappingIn3D.empty()) {
                            maxZBelow = pos.z;
                        }
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
        file.createDataSet("/data/area below", areaBelow, hdf5CreateProps);
        file.createDataSet("/data/z below", zBelow, hdf5CreateProps);

        vector<double> navRegionVec{navRegion.min.x, navRegion.min.y, navRegion.min.z,
                                    navRegion.max.x, navRegion.max.y, navRegion.max.z};
        file.createDataSet("/extra/nav region", navRegionVec);
    }

    void NavAreaAboveBelow::load(const MapMeshResult & mapMeshResult, const std::string &filePath) {
        HighFive::File file(filePath, HighFive::File::ReadOnly);

        areaAbove = file.getDataSet("/data/area above").read<std::vector<AreaId>>();
        zAbove = file.getDataSet("/data/z above").read<std::vector<float>>();
        areaBelow = file.getDataSet("/data/area below").read<std::vector<AreaId>>();
        zBelow = file.getDataSet("/data/z below").read<std::vector<float>>();
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
        std::cout << "d2 range " << navRegion.toString() << std::endl;

    }

}