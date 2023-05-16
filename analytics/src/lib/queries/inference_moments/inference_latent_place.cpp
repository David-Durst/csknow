//
// Created by durst on 4/20/23.
//

#include "queries/inference_moments/inference_latent_place.h"
#include "indices/build_indexes.h"
#include "file_helpers.h"
#include "bots/analysis/learned_models.h"

namespace csknow::inference_latent_place {
    struct LatentPlaceData {
        int64_t player, startTick;
        PlaceIndex placeIndex;
    };

    void finishPlace(vector<vector<int64_t>> & tmpStartTickId, vector<vector<int64_t>> & tmpEndTickId,
                     vector<vector<int64_t>> & tmpLength, vector<vector<int64_t>> & tmpPlayerId,
                     vector<vector<PlaceIndex>> & tmpPlaceIndex,
                     int64_t endTickIndex, int threadNum, const LatentPlaceData &lData) {
        tmpStartTickId[threadNum].push_back(lData.startTick);
        tmpEndTickId[threadNum].push_back(endTickIndex);
        tmpLength[threadNum].push_back(endTickIndex - lData.startTick + 1);
        tmpPlayerId[threadNum].push_back(lData.player);
        tmpPlaceIndex[threadNum].push_back(lData.placeIndex);
    }

    void InferenceLatentPlaceResult::runQuery(
            const string &modelsDir, const Rounds &rounds, const Ticks &ticks, const PlayerAtTick & playerAtTick,
            const csknow::behavior_tree_latent_states::BehaviorTreeLatentStates &behaviorTreeLatentStates) {
        fs::path modelPath = fs::path(modelsDir) / fs::path("latent_model") /
                             fs::path("place_script_model.pt");

        torch::NoGradGuard no_grad;
        torch::jit::getProfilingMode() = false;
        torch::jit::script::Module module;
        try {
            // Deserialize the ScriptModule from a file using torch::jit::load().
            auto tmp_module = torch::jit::load(modelPath);
            module = torch::jit::optimize_for_inference(tmp_module);
        }
        catch (const c10::Error &e) {
            size = 0;
            std::cerr << "error loading latent model\n";
            return;
        }

        std::atomic<int64_t> roundsProcessed = 0;

        int numThreads = omp_get_max_threads();
        vector<vector<int64_t>> tmpRoundIds(numThreads);
        vector<vector<int64_t>> tmpRoundStarts(numThreads);
        vector<vector<int64_t>> tmpRoundSizes(numThreads);
        vector<vector<int64_t>> tmpStartTickId(numThreads);
        vector<vector<int64_t>> tmpEndTickId(numThreads);
        vector<vector<int64_t>> tmpLength(numThreads);
        vector<vector<int64_t>> tmpPlayerId(numThreads);
        vector<vector<PlaceIndex>> tmpPlaceIndex(numThreads);

        behaviorTreeLatentStates.featureStoreResult.checkInvalid();
        playerPlaceProb.resize(playerAtTick.size);
        double inferenceSeconds = 0.;
        double numInferences = 0.;
#pragma omp parallel for
        for (int64_t roundIndex = 0; roundIndex < (runAllRounds ? rounds.size : 1L); roundIndex++) {
            int threadNum = omp_get_thread_num();
            tmpRoundIds[threadNum].push_back(roundIndex);
            tmpRoundStarts[threadNum].push_back(static_cast<int64_t>(tmpStartTickId[threadNum].size()));

            auto options = torch::TensorOptions().dtype(at::kFloat);

            map<int64_t, LatentPlaceData> playerToActivePlace;
            for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
                 tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {

                InferencePlaceTickValues values = extractFeatureStorePlaceValues(behaviorTreeLatentStates.featureStoreResult,
                                                                                 tickIndex);
                std::vector<torch::jit::IValue> inputs;
                torch::Tensor rowPT = torch::from_blob(values.rowCPP.data(), {1, static_cast<long>(values.rowCPP.size())},
                                                       options);
                inputs.push_back(rowPT);
                //std::cout << rowPT << std::endl;

                // Execute the model and turn its output into a tensor.
                auto start = std::chrono::system_clock::now();
                at::Tensor output = module.forward(inputs).toTuple()->elements()[0].toTensor();
                auto end = std::chrono::system_clock::now();
                std::chrono::duration<double> inferenceTime = end - start;
                inferenceSeconds += inferenceTime.count();
                numInferences++;
                /*
                if (ticks.demoTickNumber[tickIndex] == 5169) {
                    std::cout << "tick index " << tickIndex << " demo tick " << ticks.demoTickNumber[tickIndex] << std::endl;
                    std::cout << rowPT << std::endl;
                    for (const auto & [playerId, columnIndex] : playerIdToColumnIndex) {
                        std::cout << "player id " << playerId << " start column index " << columnIndex * total_orders << std::endl;
                    }
                }
                 */
                //std::cout << output << std::endl;
                for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                     patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                    int64_t curPlayerId = playerAtTick.playerId[patIndex];
                    TeamId curPlayerTeam = playerAtTick.team[patIndex];
                    InferencePlacePlayerAtTickProbabilities probabilities{};
                    if (playerAtTick.isAlive[patIndex]) {
                        probabilities = extractFeatureStorePlaceResults(output, values, curPlayerId, curPlayerTeam);
                        for (size_t placeIndex = 0; placeIndex < csknow::feature_store::num_places; placeIndex++) {
                            playerPlaceProb[patIndex][placeIndex] = probabilities.placeProbabilities[placeIndex];
                        }
                    }
                    /*
                    if (ticks.demoTickNumber[tickIndex] == 5169) {
                        std::cout << "player id " << curPlayerId << ", probabilities ";
                        for (size_t orderIndex = 0;
                             orderIndex < total_orders; orderIndex++) {
                            std::cout << playerOrderProb[patIndex][orderIndex] << ";";
                        }
                        std::cout << std::endl;
                    }
                     */

                    bool oldPlaceToWrite =
                            playerToActivePlace.find(curPlayerId) != playerToActivePlace.end() &&
                            (!playerAtTick.isAlive[patIndex] ||
                             playerToActivePlace[curPlayerId].placeIndex != probabilities.mostLikelyPlace);

                    if (oldPlaceToWrite) {
                        finishPlace(tmpStartTickId, tmpEndTickId,
                                    tmpLength, tmpPlayerId,
                                    tmpPlaceIndex, tickIndex,
                                    threadNum, playerToActivePlace[curPlayerId]);
                        playerToActivePlace.erase(curPlayerId);
                    }
                    if (playerToActivePlace.find(curPlayerId) == playerToActivePlace.end() &&
                        playerAtTick.isAlive[patIndex]) {
                        playerToActivePlace[curPlayerId] = {
                                curPlayerId, tickIndex, probabilities.mostLikelyPlace
                        };
                    }
                }
                //exit(0);
            }

            for (const auto & [curPlayerId, lData] : playerToActivePlace) {
                finishPlace(tmpStartTickId, tmpEndTickId,
                            tmpLength, tmpPlayerId,
                            tmpPlaceIndex, rounds.ticksPerRound[roundIndex].maxId - 1,
                            threadNum, lData);
            }

            tmpRoundSizes[threadNum].push_back(static_cast<int64_t>(tmpStartTickId[threadNum].size()) - tmpRoundStarts[threadNum].back());
            roundsProcessed++;
            printProgress(roundsProcessed, rounds.size);
        }

        mergeThreadResults(numThreads, rowIndicesPerRound, tmpRoundIds, tmpRoundStarts, tmpRoundSizes,
                           startTickId, size,
                           [&](int64_t minThreadId, int64_t tmpRowId) {
                               startTickId.push_back(tmpStartTickId[minThreadId][tmpRowId]);
                               endTickId.push_back(tmpEndTickId[minThreadId][tmpRowId]);
                               tickLength.push_back(tmpLength[minThreadId][tmpRowId]);
                               playerId.push_back(tmpPlayerId[minThreadId][tmpRowId]);
                               placeIndex.push_back(tmpPlaceIndex[minThreadId][tmpRowId]);
                           });
        vector<std::reference_wrapper<const vector<int64_t>>> foreignKeyCols{startTickId, endTickId};
        placesPerTick = buildIntervalIndex(foreignKeyCols, size);
        std::cout << "places time per inference " << inferenceSeconds / numInferences << std::endl;
    }

}
