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
        vector<vector<int64_t>> tmpPlayerId(numThreads);
        vector<vector<OrderRole>> tmpRole(numThreads);
        vector<vector<array<float, total_orders>>> tmpPlayerOrderProb(numThreads);

        behaviorTreeLatentStates.featureStoreResult.checkInvalid();
        playerOrderProb.resize(playerAtTick.size);
//#pragma omp parallel for
        for (int64_t roundIndex = 0; roundIndex < 1L /*rounds.size*/; roundIndex++) {
            int threadNum = omp_get_thread_num();
            tmpRoundIds[threadNum].push_back(roundIndex);
            tmpRoundStarts[threadNum].push_back(static_cast<int64_t>(tmpStartTickId[threadNum].size()));
            const csknow::feature_store::TeamFeatureStoreResult & teamFeatureStoreResult =
                behaviorTreeLatentStates.featureStoreResult.teamFeatureStoreResult;

            auto options = torch::TensorOptions().dtype(at::kFloat);

            map<int64_t, LatentOrderData> playerToActiveOrder;
            for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
                 tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {

                std::vector<torch::jit::IValue> inputs;
                std::vector<float> rowCPP;
                map<int64_t, size_t> playerIdToColumnIndex;
                // c4 float data
                rowCPP.push_back(static_cast<float>(teamFeatureStoreResult.c4DistanceToASite[tickIndex]));
                rowCPP.push_back(static_cast<float>(teamFeatureStoreResult.c4DistanceToBSite[tickIndex]));
                for (size_t orderIndex = 0; orderIndex < csknow::feature_store::num_orders_per_site; orderIndex++) {
                    rowCPP.push_back(static_cast<float>(
                        teamFeatureStoreResult.c4DistanceToNearestAOrderNavArea[orderIndex][tickIndex]));
                }
                for (size_t orderIndex = 0; orderIndex < csknow::feature_store::num_orders_per_site; orderIndex++) {
                    rowCPP.push_back(static_cast<float>(
                        teamFeatureStoreResult.c4DistanceToNearestBOrderNavArea[orderIndex][tickIndex]));
                }
                // player data
                bool ctColumnData = true;
                for (const auto & columnData :
                     behaviorTreeLatentStates.featureStoreResult.teamFeatureStoreResult.getAllColumnData()) {
                    for (size_t playerNum = 0; playerNum < csknow::feature_store::maxEnemies; playerNum++) {
                        const auto & columnPlayerData = columnData.get()[playerNum];
                        playerIdToColumnIndex[columnPlayerData.playerId[tickIndex]] = playerNum +
                            (ctColumnData ? 0 : csknow::feature_store::maxEnemies);
                        rowCPP.push_back(static_cast<float>(columnPlayerData.distanceToASite[tickIndex]));
                        rowCPP.push_back(static_cast<float>(columnPlayerData.distanceToBSite[tickIndex]));
                        for (size_t orderIndex = 0; orderIndex < csknow::feature_store::num_orders_per_site; orderIndex++) {
                            rowCPP.push_back(static_cast<float>(
                                columnPlayerData.distanceToNearestAOrderNavArea[orderIndex][tickIndex]));
                        }
                        for (size_t orderIndex = 0; orderIndex < csknow::feature_store::num_orders_per_site; orderIndex++) {
                            rowCPP.push_back(static_cast<float>(
                                columnPlayerData.distanceToNearestBOrderNavArea[orderIndex][tickIndex]));
                        }
                    }
                    ctColumnData = false;
                }
                // cat data
                rowCPP.push_back(static_cast<float>(teamFeatureStoreResult.c4Status[tickIndex]));

                torch::Tensor rowPT = torch::from_blob(rowCPP.data(), {1, static_cast<long>(rowCPP.size())},
                                                       options);
                inputs.push_back(rowPT);
                std::cout << rowPT << std::endl;

                // Execute the model and turn its output into a tensor.
                at::Tensor output = module.forward(inputs).toTuple()->elements()[0].toTensor();
                std::cout << output << std::endl;
                for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                     patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                    int64_t curPlayerId = playerAtTick.playerId[patIndex];
                    size_t playerStartIndex = playerIdToColumnIndex[curPlayerId] * total_orders;
                    float mostLikelyOrderProb = -1;
                    OrderRole mostLikelyOrder = OrderRole::CT0;

                    for (size_t orderIndex = 0; orderIndex < csknow::feature_store::num_orders_per_site; orderIndex++) {
                        playerOrderProb[patIndex][orderIndex] = output[0][playerStartIndex + orderIndex].item<float>();
                        if (mostLikelyOrderProb < playerOrderProb[patIndex][orderIndex]) {
                            mostLikelyOrderProb = playerOrderProb[patIndex][orderIndex];
                            mostLikelyOrder = static_cast<OrderRole>(orderIndex);
                        }
                    }
                    std::cout << "tick index " << tickIndex << ", pat index " << patIndex
                        << ", player id " << curPlayerId << ", probabilities ";
                    for (size_t orderIndex = 0; orderIndex < csknow::feature_store::num_orders_per_site; orderIndex++) {
                        std::cout << playerOrderProb[patIndex][orderIndex] << ";";
                    }
                    std::cout << std::endl;

                    bool oldOrderToWrite =
                        playerToActiveOrder.find(curPlayerId) != playerToActiveOrder.end() &&
                        playerToActiveOrder[curPlayerId].role != mostLikelyOrder;

                    if (oldOrderToWrite) {
                        finishOrder(tmpStartTickId, tmpEndTickId,
                                    tmpLength, tmpPlayerId,
                                    tmpRole, tickIndex,
                                    threadNum, playerToActiveOrder[curPlayerId]);
                        playerToActiveOrder.erase(curPlayerId);
                    }
                    if (playerToActiveOrder.find(curPlayerId) == playerToActiveOrder.end()) {
                        playerToActiveOrder[curPlayerId] = {
                            curPlayerId, tickIndex, mostLikelyOrder
                        };
                    }
                }
                exit(0);
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

    }

}