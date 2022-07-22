//
// Created by steam on 7/20/22.
//

#include "bots/save_nav_overlay.h"
#include <filesystem>

void NavFileOverlay::saveOverlays(std::stringstream & stream, Vec3 specPos, const vector<AreaBits> & overlays) {
    for (size_t i = 0; i < navAreaData.size(); i++) {
        if (computeDistance(specPos, navAreaData[i].center) > MAX_DISTANCE) {
            continue;
        }

        AABB area = {navAreaData[i].areaCoordinates.min, navAreaData[i].areaCoordinates.max};
        area.min.z = area.max.z;
        //area.min.z = area.max.z + OVERLAY_VERTICAL_BASE;
        //area.max.z = area.min.z + OVERLAY_VERTICAL_LENGTH;

        for (size_t overlayIdx = 0; overlayIdx < overlays.size(); overlayIdx++) {
            const AreaBits & overlay = overlays[overlayIdx];
            if (!overlay[i]) {
                continue;
            }
            AABB overlayArea = area;
            if (overlays.size() > 1) {
                double avgX = (overlayArea.min.x + overlayArea.max.x) / 2., avgY = (overlayArea.min.y + overlayArea.max.y) / 2.;
                if (overlayIdx == 0) {
                    overlayArea.max.x = avgX;
                    overlayArea.max.y = avgY;
                }
                else if (overlayIdx == 1) {
                    overlayArea.min.x = avgX;
                    overlayArea.max.y = avgY;
                }
                else if (overlayIdx == 2) {
                    overlayArea.max.x = avgX;
                    overlayArea.min.y = avgY;
                }
                else if (overlayIdx == 3) {
                    overlayArea.min.x = avgX;
                    overlayArea.min.y = avgY;
                }
            }
            stream << overlayArea.min.toCSV() << "," << overlayArea.max.toCSV() << "," << colorScheme[overlayIdx] << std::endl;
        }
    }
}

void NavFileOverlay::save(const ServerState & state, const vector<AreaBits> & overlays) {
    if (overlays.size() > MAX_OVERLAYS) {
        throw std::runtime_error("num overlays too large");
    }

    // drawing relative to a spectator
    bool foundSpec = false;
    size_t specIdx;
    Vec3 specPos;
    for (size_t i = 0; i < state.clients.size(); i++) {
        if (!state.clients[i].isBot && !state.clients[i].isAlive) {
            foundSpec = true;
            specIdx = i;
        }
    }
    if (foundSpec) {
        specPos = state.clients[specIdx].getFootPosForPlayer();
    }
    else {
        return;
    }

    if (state.getSecondsBetweenTimes(lastCallTime, state.loadTime) >= FADE_SECONDS) {
        lastCallTime = state.loadTime;

        string overlayFileName = "overlay.csv";
        string overlayFilePath = state.dataPath + "/" + overlayFileName;
        string tmpOverlayFileName = "overlay.csv.tmp.write";
        string tmpOverlayFilePath = state.dataPath + "/" + tmpOverlayFileName;

        std::stringstream overlayStream;

        overlayStream << FADE_SECONDS << std::endl;



        saveOverlays(overlayStream, specPos, overlays);

        std::ofstream fsOverlay(tmpOverlayFilePath);
        fsOverlay << overlayStream.str();
        fsOverlay.close();

        std::filesystem::rename(tmpOverlayFilePath, overlayFilePath);

        int x = 1;

    }
}
