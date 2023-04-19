//
// Created by durst on 3/6/23.
//

#include "queries/inference_moments/inference_latent_engagement.h"
#include "file_helpers.h"
#include "indices/build_indexes.h"
#include "bots/analysis/learned_models.h"

namespace csknow::inference_latent_engagement {

    struct LatentEngagementData {
        int64_t attacker, victim;
        int64_t startTick;
        vector<int64_t> hurtTickIds, hurtIds;
    };

    void finishEngagement(vector<vector<int64_t>> & tmpStartTickId, vector<vector<int64_t>> & tmpEndTickId,
                          vector<vector<int64_t>> & tmpLength, vector<vector<vector<int64_t>>> & tmpPlayerId,
                          vector<vector<vector<EngagementRole>>> & tmpRole,
                          vector<vector<vector<int64_t>>> & tmpHurtTickIds, vector<vector<vector<int64_t>>> & tmpHurtIds,
                          int64_t endTickIndex, int threadNum, const LatentEngagementData &eData) {
        tmpStartTickId[threadNum].push_back(eData.startTick);
        tmpEndTickId[threadNum].push_back(endTickIndex);
        tmpLength[threadNum].push_back(endTickIndex - eData.startTick + 1);
        tmpPlayerId[threadNum].push_back({eData.attacker, eData.victim});
        tmpRole[threadNum].push_back({EngagementRole::Attacker, EngagementRole::Victim});
        tmpHurtTickIds[threadNum].push_back(eData.hurtTickIds);
        tmpHurtIds[threadNum].push_back(eData.hurtIds);
    }

    void InferenceLatentEngagementResult::runQuery(
        const string & modelsDir, const Rounds & rounds, const Ticks & ticks,
        const csknow::behavior_tree_latent_states::BehaviorTreeLatentStates & behaviorTreeLatentStates) {
        fs::path modelPath = fs::path(modelsDir) / fs::path("latent_model") /
                             fs::path("engagement_script_model.pt");

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
        vector<vector<vector<EngagementRole>>> tmpRole(numThreads);
        vector<vector<vector<int64_t>>> tmpHurtTickIds(numThreads);
        vector<vector<vector<int64_t>>> tmpHurtIds(numThreads);
        vector<vector<vector<PlayerEngageProb>>> tmpPlayerEngageProb(numThreads);

        behaviorTreeLatentStates.featureStoreResult.checkInvalid();
        playerEngageProbs.resize(playerAtTick.size);
#pragma omp parallel for
        for (int64_t roundIndex = 0; roundIndex < (runAllRounds ? rounds.size : 1L); roundIndex++) {
            int threadNum = omp_get_thread_num();
            tmpRoundIds[threadNum].push_back(roundIndex);
            tmpRoundStarts[threadNum].push_back(static_cast<int64_t>(tmpStartTickId[threadNum].size()));

            auto options = torch::TensorOptions().dtype(at::kFloat);

            map<int64_t, LatentEngagementData> playerToActiveEngagement;
            for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
                 tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {
                if (tickIndex % 10 != 0) {
                    continue;
                }
                for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                     patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                    int64_t curPlayerId = playerAtTick.playerId[patIndex];

                    InferenceEngagementTickValues values =
                        extractFeatureStoreEngagementValues(behaviorTreeLatentStates.featureStoreResult, patIndex);
                    std::vector<torch::jit::IValue> inputs;

                    torch::Tensor rowPT = torch::from_blob(values.rowCPP.data(), {1, static_cast<long>(values.rowCPP.size())},
                                                           options);
                    inputs.push_back(rowPT);

                    // Execute the model and turn its output into a tensor.
                    at::Tensor output = module.forward(inputs).toTuple()->elements()[0].toTensor();

                    /*
                    if (tickIndex == 8080 && curPlayerId == 3) {
                        std::cout << "tick index " << tickIndex << " cur player id " << curPlayerId
                            << " game tick " <<  ticks.gameTickNumber[tickIndex]
                            << " demo tick " << ticks.demoTickNumber[tickIndex] << std::endl;
                        for (size_t enemyNum = 0; enemyNum < csknow::feature_store::maxEnemies; enemyNum++) {
                            std::cout << "cur player " << curPlayerId
                                << " enemy num " << enemyNum
                                << " player id " << behaviorTreeLatentStates.featureStoreResult.columnEnemyData[enemyNum].playerId[patIndex]
                                << " world distance " << behaviorTreeLatentStates.featureStoreResult.columnEnemyData[enemyNum].worldDistanceToEnemy[patIndex]
                                << " crosshair distance " << behaviorTreeLatentStates.featureStoreResult.columnEnemyData[enemyNum].crosshairDistanceToEnemy[patIndex]
                                << " enemy prob " << output[0][enemyNum].item<float>()
                                << std::endl;
                        }
                    }
                     */
                    // for distribution visualization
                    for (size_t enemyNum = 0; enemyNum <= csknow::feature_store::maxEnemies; enemyNum++) {
                        int64_t enemyId = INVALID_ID;
                        if (enemyNum < csknow::feature_store::maxEnemies) {
                            enemyId = behaviorTreeLatentStates.featureStoreResult.columnEnemyData[enemyNum].playerId[patIndex];
                        }
                        playerEngageProbs[patIndex][enemyNum] = {enemyId, output[0][enemyNum].item<float>()};
                    }
                    InferenceEngagementTickProbabilities probabilities =
                        extractFeatureStoreEngagementResults(
                            //behaviorTreeLatentStates.featureStoreResult, patIndex, tickIndex,
                            output, values);

                    /*
                    bool hitEngagement = output[0][csknow::feature_store::maxEnemies * 2].item<float>() >= 0.5;
                    if (hitEngagement) {
                        std::cout << "hit engagement" << std::endl;
                    }
                    bool visibleEngagement = output[0][csknow::feature_store::maxEnemies * 2 + 1].item<float>() >= 0.5;
                    if (visibleEngagement) {
                        std::cout << "visible engagement" << std::endl;
                    }
                     */

                    bool engagement = probabilities.mostLikelyEnemyNum < csknow::feature_store::maxEnemies;
                    int firstLikelyEnemy = INVALID_ID;
                    if (engagement) {
                        /*
                        for (size_t enemyNum = 0; enemyNum <= csknow::feature_store::maxEnemies; enemyNum++) {
                            std::cout << output[0][enemyNum].item<float>() << std::endl;
                        }
                         */
                        firstLikelyEnemy = behaviorTreeLatentStates.featureStoreResult
                            .columnEnemyData[probabilities.mostLikelyEnemyNum].playerId[patIndex];
                        //exit(1);
                    }
                    /*
                    if (engagement && firstLikelyEnemy == INVALID_ID) {
                        std::cout << "bad" << std::endl;
                        std::cout << "most likely enemy num: " << mostLikelyEnemyNum << std::endl;
                        for (size_t enemyNum = 0; enemyNum < csknow::feature_store::maxEnemies; enemyNum++) {
                            const csknow::feature_store::FeatureStoreResult::ColumnEnemyData &columnEnemyData =
                                behaviorTreeLatentStates.featureStoreResult.columnEnemyData[enemyNum];
                            std::cout << "enemyNum " << enemyNum << ", player id " << columnEnemyData.playerId[patIndex] <<
                                ", engagement state " << enumAsInt(columnEnemyData.enemyEngagementStates[patIndex]) <<
                                ", prob " << enemyProbabilities[enemyNum] << " other prob " << output[0][enemyNum].item<float>() << std::endl;


                        }
                    }
                     */

                    bool oldEngagementToWrite =
                        // if was engagement and now none
                        (!engagement &&
                         playerToActiveEngagement.find(curPlayerId) != playerToActiveEngagement.end()) ||
                        // new engagmenet with different target
                        (engagement &&
                         playerToActiveEngagement.find(curPlayerId) != playerToActiveEngagement.end() &&
                         playerToActiveEngagement[curPlayerId].victim != firstLikelyEnemy);

                    // if new engagement and no old engagement, just add to tracker
                    if (oldEngagementToWrite) {
                        //std::cout << "writing latent engagement" << std::endl;
                        finishEngagement(tmpStartTickId, tmpEndTickId,
                                         tmpLength, tmpPlayerId,
                                         tmpRole,
                                         tmpHurtTickIds, tmpHurtIds, tickIndex,
                                         threadNum, playerToActiveEngagement[curPlayerId]);
                        playerToActiveEngagement.erase(curPlayerId);
                    }
                    if (engagement && playerToActiveEngagement.find(curPlayerId) == playerToActiveEngagement.end()) {
                        //std::cout << "starting latent engagement" << std::endl;
                        playerToActiveEngagement[curPlayerId] = {
                            curPlayerId, firstLikelyEnemy,
                            tickIndex, vector<int64_t>(), vector<int64_t>()
                        };
                    }
                }
            }

            for (const auto & [curPlayerId, eData] : playerToActiveEngagement) {
                finishEngagement(tmpStartTickId, tmpEndTickId,
                                 tmpLength, tmpPlayerId,
                                 tmpRole,
                                 tmpHurtTickIds, tmpHurtIds, rounds.ticksPerRound[roundIndex].maxId - 1,
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
                               hurtTickIds.push_back(tmpHurtTickIds[minThreadId][tmpRowId]);
                               hurtIds.push_back(tmpHurtIds[minThreadId][tmpRowId]);
                           });
        vector<const int64_t *> foreignKeyCols{startTickId.data(), endTickId.data()};
        engagementsPerTick = buildIntervalIndex(foreignKeyCols, size);
    }
}