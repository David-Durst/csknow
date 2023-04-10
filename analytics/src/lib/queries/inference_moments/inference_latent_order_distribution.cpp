//
// Created by durst on 4/10/23.
//

#include "queries/inference_moments/inference_latent_order_distribution.h"

namespace csknow::inference_latent_order {
    void InferenceLatentOrderDistributionResult::setLabelPositions(const MapMeshResult & mapMeshResult) {
        const map<PlaceIndex, vector<AreaId>> & placesToAreas = ordersResult.visPoints.getPlacesToAreas();
        for (size_t orderIndex = 0; orderIndex < ordersResult.orders.size(); orderIndex++) {
            PlaceIndex orderPlaceIndex = ordersResult.orders[orderIndex].places[orderToPlace[orderIndex]];
            AABB placeBounds {
                {
                    std::numeric_limits<double>::max(),
                    std::numeric_limits<double>::max(),
                    std::numeric_limits<double>::max(),
                },
                {
                    -1 * std::numeric_limits<double>::max(),
                    -1 * std::numeric_limits<double>::max(),
                    -1 * std::numeric_limits<double>::max(),
                }
            };
            for (const auto & areaId : placesToAreas.at(orderPlaceIndex)) {
                int64_t areaIndex = mapMeshResult.areaToInternalId.at(areaId);
                placeBounds.min = min(placeBounds.min, mapMeshResult.coordinate[areaIndex].min);
                placeBounds.max = max(placeBounds.max, mapMeshResult.coordinate[areaIndex].max);
            }
            orderLabelPositions[orderIndex] = getCenter(placeBounds);
        }
    }
}
