//
// Created by durst on 3/6/23.
//

#include "queries/inference_moments/inference_latent_aggression.h"
#include "file_helpers.h"
#include "indices/build_indexes.h"

namespace csknow::inference_latent_aggression {

    struct LatentAggressionData {
        int64_t attacker;
        int64_t startTick;
        feature_store::NearestEnemyState aggressionState;
    };

    void finishAggression(vector<vector<int64_t>> & tmpStartTickId, vector<vector<int64_t>> & tmpEndTickId,
                          vector<vector<int64_t>> & tmpLength, vector<vector<vector<int64_t>>> & tmpPlayerId,
                          vector<vector<vector<feature_store::NearestEnemyState>>> & tmpRole,
                          int64_t endTickIndex, int threadNum, const LatentAggressionData &eData) {
        tmpStartTickId[threadNum].push_back(eData.startTick);
        tmpEndTickId[threadNum].push_back(endTickIndex);
        tmpLength[threadNum].push_back(endTickIndex - eData.startTick + 1);
        tmpPlayerId[threadNum].push_back({eData.attacker});
        tmpRole[threadNum].push_back({eData.aggressionState});
    }

    void InferenceLatentAggressionResult::runQuery(
        const string & modelsDir, const Rounds & rounds, const Ticks & ticks,
        const csknow::behavior_tree_latent_states::BehaviorTreeLatentStates & behaviorTreeLatentStates) {
        fs::path modelPath = fs::path(modelsDir) / fs::path("latent_model") /
                             fs::path("aggression_script_model.pt");

        torch::jit::getProfilingMode() = false;
        torch::jit::script::Module module;
        try {
            // Deserialize the ScriptModule from a file using torch::jit::load().
            module = torch::jit::load(modelPath);
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
        vector<vector<vector<int64_t>>> tmpPlayerId(numThreads);
        vector<vector<vector<feature_store::NearestEnemyState>>> tmpRole(numThreads);

        behaviorTreeLatentStates.featureStoreResult.checkInvalid();
        playerAggressionProb.resize(playerAtTick.size);
//#pragma omp parallel for
        for (int64_t roundIndex = 0; roundIndex < 1L /*rounds.size*/; roundIndex++) {
            int threadNum = omp_get_thread_num();
            tmpRoundIds[threadNum].push_back(roundIndex);
            tmpRoundStarts[threadNum].push_back(static_cast<int64_t>(tmpStartTickId[threadNum].size()));

            auto options = torch::TensorOptions().dtype(at::kFloat);

            map<int64_t, LatentAggressionData> playerToActiveAggression;
            for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
                 tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {
                for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                     patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                    int64_t curPlayerId = playerAtTick.playerId[patIndex];

                    InferenceAggressionTickValues values =
                        extractFeatureStoreAggressionValues(behaviorTreeLatentStates.featureStoreResult, patIndex);
                    std::vector<torch::jit::IValue> inputs;
                    torch::Tensor rowPT = torch::from_blob(values.rowCPP.data(), {1, static_cast<long>(values.rowCPP.size())},
                                                           options);
                    inputs.push_back(rowPT);

                    // Execute the model and turn its output into a tensor.
                    at::Tensor output = module.forward(inputs).toTuple()->elements()[0].toTensor();

                    for (size_t aggressionOption = 0; aggressionOption < csknow::feature_store::numNearestEnemyState;
                         aggressionOption++) {
                        //std::cout << output[0][enemyNum].item<float>() << std::endl;
                        playerAggressionProb[patIndex][aggressionOption] = output[0][aggressionOption].item<float>();
                    }
                    InferenceAggressionTickProbabilities probabilities =
                        extractFeatureStoreAggressionResults(output, values);

                    bool oldAggressionToWrite =
                        playerToActiveAggression.find(curPlayerId) != playerToActiveAggression.end() &&
                        playerToActiveAggression[curPlayerId].aggressionState != probabilities.mostLikelyAggression;

                    // if new engagement and no old engagement, just add to tracker
                    if (oldAggressionToWrite) {
                        //std::cout << "writing latent aggression" << std::endl;
                        finishAggression(tmpStartTickId, tmpEndTickId,
                                         tmpLength, tmpPlayerId,
                                         tmpRole, tickIndex,
                                         threadNum, playerToActiveAggression[curPlayerId]);
                        playerToActiveAggression.erase(curPlayerId);
                    }
                    if (playerToActiveAggression.find(curPlayerId) == playerToActiveAggression.end()) {
                        //std::cout << "starting latent aggression" << std::endl;
                        playerToActiveAggression[curPlayerId] = {
                            curPlayerId, tickIndex, probabilities.mostLikelyAggression
                        };
                    }
                }
            }

            for (const auto & [curPlayerId, eData] : playerToActiveAggression) {
                finishAggression(tmpStartTickId, tmpEndTickId,
                                 tmpLength, tmpPlayerId,
                                 tmpRole,
                                 rounds.ticksPerRound[roundIndex].maxId - 1,
                                 threadNum, eData);
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
                               role.push_back(tmpRole[minThreadId][tmpRowId]);
                           });
        vector<const int64_t *> foreignKeyCols{startTickId.data(), endTickId.data()};
        aggressionPerTick = buildIntervalIndex(foreignKeyCols, size);
    }
}