//
// Created by steam on 7/20/22.
//

#include "bots/save_nav_overlay.h"
#include <filesystem>

void NavFileOverlay::saveOverlay(std::stringstream & stream, size_t overlayIndex, size_t numOverlays,
                                 const map<CSGOId, AreaBits> & playerToOverlay) {
    for (size_t i = 0; i < navAreaData.size(); i++) {
        AABB area = {navAreaData[i].areaCoordinates.min, navAreaData[i].areaCoordinates.max};
        area.min.z = area.max.z + OVERLAY_VERTICAL_BASE;
        area.max.z = area.min.z + OVERLAY_VERTICAL_LENGTH;
        if (numOverlays > 2) {
            throw std::runtime_error("num overlays too large");
        }
        if (numOverlays == 2) {
            // divide areas in half if two overlays at once
            if (overlayIndex == 0) {
                area.max.x = (area.min.x + area.max.x) / 2;
            }
            else if (overlayIndex == 1) {
                area.min.x = (area.min.x + area.max.x) / 2;
            }
            else {
                throw std::runtime_error("too large overlay index at once");
            }
        }

        size_t playerIndex = 0;
        for (const auto & [_, areaBits] : playerToOverlay) {
            AABB playerArea = area;
            if (playerToOverlay.size() > 1) {
                double avgX = (playerArea.min.x + playerArea.max.x) / 2., avgY = (playerArea.min.y + playerArea.max.y) / 2.;
                if (playerIndex == 0) {
                    playerArea.max.x = avgX;
                    playerArea.max.y = avgY;
                }
                else if (playerIndex == 1) {
                    playerArea.min.x = avgX;
                    playerArea.max.y = avgY;
                }
                else if (playerIndex == 2) {
                    playerArea.max.x = avgX;
                    playerArea.min.y = avgY;
                }
                else if (playerIndex == 3) {
                    playerArea.min.x = avgX;
                    playerArea.min.y = avgY;
                }
                else {
                    throw std::runtime_error("max players per overlay is 4, attempting to add player index " + std::to_string(playerIndex) + " for overlay");
                }
            }
            stream << playerArea.min.toCSV() << "," << playerArea.max.toCSV() << "," << colorScheme[overlayIndex * MAX_PLAYERS_PER_OVERLAY + playerIndex] << std::endl;
            playerIndex++;
        }
    }
}

void NavFileOverlay::save(const ServerState & state, const vector<map<CSGOId, AreaBits>> & overlays) {
    if (state.getSecondsBetweenTimes(lastCallTime, state.loadTime) >= FADE_SECONDS) {
        lastCallTime = state.loadTime;

        string overlayFileName = "overlay.csv";
        string overlayFilePath = state.dataPath + "/" + overlayFileName;
        string tmpOverlayFileName = "overlay.csv.tmp.write";
        string tmpOverlayFilePath = state.dataPath + "/" + tmpOverlayFileName;

        std::stringstream overlayStream;

        overlayStream << FADE_SECONDS << std::endl;

        size_t overlayIndex = 0;
        for (const auto & playerToOverlay : overlays) {
            saveOverlay(overlayStream, overlayIndex, overlays.size(), playerToOverlay);
            overlayIndex++;
        }

        std::ofstream fsOverlay(tmpOverlayFilePath);
        fsOverlay << overlayStream.str();
        fsOverlay.close();

        std::filesystem::rename(tmpOverlayFilePath, overlayFilePath);

    }
}
