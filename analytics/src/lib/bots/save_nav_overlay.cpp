//
// Created by steam on 7/20/22.
//

#include "bots/save_nav_overlay.h"

void PlayerNavAreaOverlay::saveOverlay(std::ofstream & stream, size_t playerIndex, size_t overlayIndex,
                                       size_t numOverlays, const vector<NavAreaData> & navAreaData) {
    double verticalMin = OVERLAY_VERTICAL_BASE + overlayIndex * OVERLAY_VERTICAL_OFFSET;
    for (size_t i = 0; i < navAreaData.size(); i++) {
        if (playerToOverlay[])
        if (numOverlays == 1) {
            AABB area = {navAreaData[i].areaCoordinates.max, navAreaData[i].areaCoordinates.max};
            area.min.z += verticalMin;
            area.max.z = area.min.z + OVERLAY_VERTICAL_LENGTH;
            stream << area.min.toCSV() << "," << area.max.toCSV() << "," << colorScheme[0] << std::endl;
        }
        else {
            AABB area = {navAreaData[i].areaCoordinates.max, navAreaData[i].areaCoordinates.max};
            area.min.z += verticalMin;
            area.max.z = area.min.z + OVERLAY_VERTICAL_LENGTH;
            double avgX = (area.min.x + area.max.x) / 2., avgY = (area.min.y + area.max.y) / 2.;
            if (playerIndex == 0) {
                area.max.x = avgX;
                area.max.y = avgY;
            }
            else if (playerIndex == 1) {
                area.min.x = avgX;
                area.max.y = avgY;
            }
            else if (playerIndex == 2) {
                area.max.x = avgX;
                area.min.y = avgY;
            }
            else if (playerIndex == 3) {
                area.min.x = avgX;
                area.min.y = avgY;
            }
            else {
                throw std::runtime_error("max players per overlay is 4, attempting to add player index " + std::to_string(playerIndex) + " for overlay");
            }
            stream << area.min.toCSV() << "," << area.max.toCSV() << "," << colorScheme[playerIndex + 1] << std::endl;
        }
    }
}
