//
// Created by durst on 10/2/22.
//

#include "queries/nav_cells.h"

MapCellsResult queryMapCells(const VisPoints & visPoints, const nav_mesh::nav_file & navFile, const string & queryName) {
    MapCellsResult result(queryName);
    for (const auto & navCell : visPoints.getCellVisPoints()) {
        result.id.push_back(navCell.cellId);
        result.areaId.push_back(navCell.areaId);
        const auto & navArea = navFile.get_area_by_id_fast(navCell.areaId);
        if (navArea.m_place < navFile.m_place_count) {
            result.placeName.push_back(navFile.get_place(navArea.m_place));
        }
        else {
            result.placeName.push_back("");
        }
        result.coordinate.push_back(navCell.cellCoordinates);
        result.connectionAreaIds.push_back({});
    }
    return result;
}
