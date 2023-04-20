//
// Created by durst on 4/20/23.
//

#include "queries/inference_moments/inference_latent_place_distribution.h"

namespace csknow::inference_latent_place {
    void InferenceLatentPlaceDistributionResult::setLabelPositions() {
        for (size_t i = 0; i < distanceToPlacesResult.places.size(); i++) {
            const string & placeName = distanceToPlacesResult.places[i];
            const AABB & placeAABB = distanceToPlacesResult.placeToAABB.at(placeName);
            placeLabelPositions[i] = getCenter(placeAABB);
        }
    }
}
