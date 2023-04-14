//
// Created by durst on 4/9/23.
//

#include "queries/inference_moments/inference_latent_order.h"
#include "indices/build_indexes.h"
#include "file_helpers.h"

namespace csknow::inference_latent_order {
    struct LatentOrderData {
        int64_t player, startTick;
        OrderRole role;
    };

    void finishOrder(vector<vector<int64_t>> & tmpStartTickId, vector<vector<int64_t>> & tmpEndTickId,
                          vector<vector<int64_t>> & tmpLength, vector<vector<int64_t>> & tmpPlayerId,
                          vector<vector<OrderRole>> & tmpRole,
                          int64_t endTickIndex, int threadNum, const LatentOrderData &lData) {
        tmpStartTickId[threadNum].push_back(lData.startTick);
        tmpEndTickId[threadNum].push_back(endTickIndex);
        tmpLength[threadNum].push_back(endTickIndex - lData.startTick + 1);
        tmpPlayerId[threadNum].push_back(lData.player);
        tmpRole[threadNum].push_back(lData.role);
    }

    void InferenceLatentOrderResult::runQuery(
        const string &modelsDir, const Rounds &rounds, const Ticks &ticks, const PlayerAtTick & playerAtTick,
        const csknow::behavior_tree_latent_states::BehaviorTreeLatentStates &behaviorTreeLatentStates) {
        fs::path modelPath = fs::path(modelsDir) / fs::path("latent_model") /
                             fs::path("order_script_model.pt");

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
        vector<vector<OrderRole>> tmpRole(numThreads);
        vector<vector<array<float, total_orders>>> tmpPlayerOrderProb(numThreads);

        behaviorTreeLatentStates.featureStoreResult.checkInvalid();
        playerOrderProb.resize(playerAtTick.size);
        double inferenceSeconds = 0.;
        double numInferences = 0.;
//#pragma omp parallel for
        for (int64_t roundIndex = 0; roundIndex < 1L /*rounds.size*/; roundIndex++) {
            int threadNum = omp_get_thread_num();
            tmpRoundIds[threadNum].push_back(roundIndex);
            tmpRoundStarts[threadNum].push_back(static_cast<int64_t>(tmpStartTickId[threadNum].size()));

            auto options = torch::TensorOptions().dtype(at::kFloat);

            map<int64_t, LatentOrderData> playerToActiveOrder;
            for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
                 tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {

                InferenceOrderTickValues values = extractFeatureStoreOrderValues(behaviorTreeLatentStates.featureStoreResult,
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
                    InferenceOrderPlayerAtTickProbabilities probabilities{};
                    if (playerAtTick.isAlive[patIndex]) {
                        probabilities = extractFeatureStoreOrderResults(output, values, curPlayerId);
                        for (size_t orderIndex = 0; orderIndex < total_orders; orderIndex++) {
                            playerOrderProb[patIndex][orderIndex] = probabilities.orderProbabilities[orderIndex];
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

                    bool oldOrderToWrite =
                        playerToActiveOrder.find(curPlayerId) != playerToActiveOrder.end() &&
                        (!playerAtTick.isAlive[patIndex] ||
                         playerToActiveOrder[curPlayerId].role != probabilities.mostLikelyOrder);

                    if (oldOrderToWrite) {
                        finishOrder(tmpStartTickId, tmpEndTickId,
                                    tmpLength, tmpPlayerId,
                                    tmpRole, tickIndex,
                                    threadNum, playerToActiveOrder[curPlayerId]);
                        playerToActiveOrder.erase(curPlayerId);
                    }
                    if (playerToActiveOrder.find(curPlayerId) == playerToActiveOrder.end() &&
                        playerAtTick.isAlive[patIndex]) {
                        playerToActiveOrder[curPlayerId] = {
                            curPlayerId, tickIndex, probabilities.mostLikelyOrder
                        };
                    }
                }
                //exit(0);
            }

            for (const auto & [curPlayerId, lData] : playerToActiveOrder) {
                finishOrder(tmpStartTickId, tmpEndTickId,
                            tmpLength, tmpPlayerId,
                            tmpRole, rounds.ticksPerRound[roundIndex].maxId - 1,
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
                               role.push_back(tmpRole[minThreadId][tmpRowId]);
                           });
        vector<const int64_t *> foreignKeyCols{startTickId.data(), endTickId.data()};
        ordersPerTick = buildIntervalIndex(foreignKeyCols, size);
        std::cout << "orders time per inference " << inferenceSeconds / numInferences << std::endl;
    }

}