//
// Created by steam on 7/20/22.
//

#include "bots/save_nav_overlay.h"
#include <filesystem>

struct NavAreaDistance {
    size_t index;
    double distanceToSpecPos;
};

void NavFileOverlay::saveOverlays(std::stringstream & stream, Vec3 specPos, const vector<AreaBits> & overlays) {
    vector<NavAreaDistance> sortedData;
    for (size_t i = 0; i < navAreaData.size(); i++) {
        sortedData.push_back({i, computeDistance(specPos, navAreaData[i].center)});
    }
    std::sort(sortedData.begin(), sortedData.end(), [](const NavAreaDistance & a, const NavAreaDistance & b) { return a.distanceToSpecPos > b.distanceToSpecPos; });

    for (size_t i = 0; i < navAreaData.size(); i++) {
        NavAreaData oneAreaData = navAreaData[sortedData[i].index];
        if (computeDistance(specPos, oneAreaData.center) > MAX_DISTANCE) {
            continue;
        }

        AABB area = {oneAreaData.areaCoordinates.min, oneAreaData.areaCoordinates.max};
        area.min.z = area.max.z;

        size_t validOverlaysNumber = 0;
        for (size_t overlayIdx = 0; overlayIdx < overlays.size(); overlayIdx++) {
            if (overlays[overlayIdx][sortedData[i].index]) {
                validOverlaysNumber += 1 << overlayIdx;
            }
        }
        stream << area.min.toCSV() << "," << area.max.toCSV() << "," << validOverlaysNumber << std::endl;
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
