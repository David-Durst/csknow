//
// Created by durst on 3/6/23.
//

#include "queries/inference_moments/inference_latent_engagement.h"
#include "file_helpers.h"

namespace csknow::inference_latent_engagement {

    struct LatentEngagementData {
        int64_t attacker, victim;
        int64_t startTick, endTick = INVALID_ID;
        vector<int64_t> hurtTickIds, hurtIds;
    };

    void finishEngagement(vector<vector<int64_t>> & tmpStartTickId, vector<vector<int64_t>> & tmpEndTickId,
                          vector<vector<int64_t>> & tmpLength, vector<vector<vector<int64_t>>> & tmpPlayerId,
                          vector<vector<vector<EngagementRole>>> & tmpRole,
                          vector<vector<vector<int64_t>>> & tmpHurtTickIds, vector<vector<vector<int64_t>>> & tmpHurtIds,
                          int threadNum, const LatentEngagementData &eData) {
        tmpStartTickId[threadNum].push_back(eData.startTick);
        tmpEndTickId[threadNum].push_back(eData.endTick);
        tmpLength[threadNum].push_back(eData.endTick - eData.startTick + 1);
        tmpPlayerId[threadNum].push_back({eData.attacker, eData.victim});
        tmpRole[threadNum].push_back({EngagementRole::Attacker, EngagementRole::Victim});
        tmpHurtTickIds[threadNum].push_back(eData.hurtTickIds);
        tmpHurtIds[threadNum].push_back(eData.hurtIds);
    }

    void InferenceLatentEngagementResult::runQuery(
        const string & modelsDir, const Rounds & rounds, const Ticks & ticks,
        const csknow::behavior_tree_latent_states::BehaviorTreeLatentStates & behaviorTreeLatentStates) {
        fs::path modelPath = fs::path(modelsDir) / fs::path("latent_model") /
                             fs::path("script_model.pt");

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
        vector<vector<vector<EngagementRole>>> tmpRole(numThreads);
        vector<vector<vector<int64_t>>> tmpHurtTickIds(numThreads);
        vector<vector<vector<int64_t>>> tmpHurtIds(numThreads);

//#pragma omp parallel for
        for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
            int threadNum = omp_get_thread_num();
            auto options = torch::TensorOptions().dtype(at::kFloat);
            map<int64_t, LatentEngagementData> playerToActiveEngagement;
            for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
                 tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {
                for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                     patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                    int64_t curPlayerId = playerAtTick.playerId[patIndex];
                    std::vector<torch::jit::IValue> inputs;
                    std::vector<float> rowCPP;
                    // all but cur tick are inputs
                    // seperate different input types
                    for (size_t enemyNum = 0; enemyNum < csknow::feature_store::maxEnemies; enemyNum++) {
                        const csknow::feature_store::FeatureStoreResult::ColumnEnemyData &columnEnemyData =
                            behaviorTreeLatentStates.featureStoreResult.columnEnemyData[enemyNum];
                        rowCPP.push_back(static_cast<float>(columnEnemyData.playerId[patIndex]));
                        rowCPP.push_back(static_cast<float>(columnEnemyData.enemyEngagementStates[patIndex]));
                        rowCPP.push_back(
                            static_cast<float>(columnEnemyData.timeSinceLastVisibleOrToBecomeVisible[patIndex]));
                        rowCPP.push_back(static_cast<float>(columnEnemyData.worldDistanceToEnemy[patIndex]));
                        rowCPP.push_back(static_cast<float>(columnEnemyData.crosshairDistanceToEnemy[patIndex]));
                    }

                    torch::Tensor rowPT = torch::from_blob(rowCPP.data(), {1, static_cast<long>(rowCPP.size())},
                                                           options);
                    inputs.push_back(rowPT);

                    // Execute the model and turn its output into a tensor.
                    at::Tensor output = module.forward(inputs).toTuple()->elements()[1].toTensor();

                    vector<csknow::feature_store::TargetPossibleEnemyLabel> targetLabels;
                    int firstLikelyHitEnemy = INVALID_ID;
                    for (size_t enemyNum = 0; enemyNum < csknow::feature_store::maxEnemies; enemyNum++) {
                        const csknow::feature_store::FeatureStoreResult::ColumnEnemyData &columnEnemyData =
                            behaviorTreeLatentStates.featureStoreResult.columnEnemyData[enemyNum];
                        targetLabels.push_back({
                                                   columnEnemyData.playerId[patIndex],
                                                   output[0][enemyNum * 2].item<float>() >= 0.5,
                                                   output[0][enemyNum * 2 + 1].item<float>() >= 0.5
                                               });
                        if (firstLikelyHitEnemy == INVALID_ID && targetLabels.back().hitTargetEnemy) {
                            firstLikelyHitEnemy = columnEnemyData.playerId[patIndex];
                        }
                    }

                    bool hitEngagement = output[0][csknow::feature_store::maxEnemies * 2].item<float>() >= 0.5;

                    bool oldEngagementToWrite =
                        // if was engagement and now none
                        (!hitEngagement &&
                         playerToActiveEngagement.find(curPlayerId) != playerToActiveEngagement.end()) ||
                        // new engagmenet with different target
                        (hitEngagement &&
                         playerToActiveEngagement.find(curPlayerId) != playerToActiveEngagement.end() &&
                         playerToActiveEngagement[curPlayerId].victim != firstLikelyHitEnemy);

                    // if new engagement and no old engagement, just add to tracker
                    if (oldEngagementToWrite) {
                        finishEngagement(tmpStartTickId, tmpEndTickId,
                                         tmpLength, tmpPlayerId,
                                         tmpRole,
                                         tmpHurtTickIds, tmpHurtIds,
                                         threadNum, playerToActiveEngagement[curPlayerId]);
                        playerToActiveEngagement.erase(curPlayerId);
                    }
                    if (hitEngagement && oldEngagementToWrite) {
                        playerToActiveEngagement[curPlayerId] = {
                            curPlayerId, firstLikelyHitEnemy,
                            tickIndex, INVALID_ID, vector<int64_t>(), vector<int64_t>()
                        };
                    }
                }
            }

            for (const auto & [curPlayerId, eData] : playerToActiveEngagement) {
                finishEngagement(tmpStartTickId, tmpEndTickId,
                                 tmpLength, tmpPlayerId,
                                 tmpRole,
                                 tmpHurtTickIds, tmpHurtIds,
                                 threadNum, eData);
            }

            roundsProcessed++;
            printProgress(roundsProcessed, rounds.size);
        }
        size = playerAtTick.size;
    }
}