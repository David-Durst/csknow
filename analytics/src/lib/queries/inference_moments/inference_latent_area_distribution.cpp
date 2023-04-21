//
// Created by durst on 4/21/23.
//

#include "queries/inference_moments/inference_latent_area_distribution.h"

namespace csknow::inference_latent_area {
    void InferenceLatentAreaDistributionResult::computeMostLikelyPlace() {
        mostLikelyPlace.resize(playerAtTick.size);
        for (int64_t patIndex = 0; patIndex < playerAtTick.size; patIndex++) {
            PlaceIndex maxProbPlaceIndex = 0;
            if (playerAtTick.isAlive[patIndex]) {
                double maxProb = -1 * std::numeric_limits<double>::max();
                for (PlaceIndex curPlaceIndex = 0; curPlaceIndex < distanceToPlacesResult.places.size(); curPlaceIndex++) {
                    double curProb = inferenceLatentPlaceResult.playerPlaceProb[patIndex][curPlaceIndex];
                    if (curProb > maxProb) {
                        maxProb = curProb;
                        maxProbPlaceIndex = curPlaceIndex;
                    }
                }
            }
            mostLikelyPlace[patIndex] = maxProbPlaceIndex;
        }
    }
}
