//
// Created by durst on 4/9/23.
//

#ifndef CSKNOW_INFERENCE_LATENT_ORDER_DISTRIBUTION_H
#define CSKNOW_INFERENCE_LATENT_ORDER_DISTRIBUTION_H

#include "queries/inference_moments/inference_latent_order.h"
#include "queries/base_tables.h"

namespace csknow::inference_latent_order {
    class InferenceLatentOrderDistributionResult : public QueryResult {
    public:
        const PlayerAtTick & playerAtTick;
        QueryPlayerAtTick & queryPlayerAtTick;
        const orders::OrdersResult & ordersResult;
        const InferenceLatentOrderResult & inferenceLatentOrderResult;
        array<PlaceIndex, total_orders> orderToPlace{4, 2, 3, 5, 4, 2};
        array<Vec3, total_orders> orderLabelPositions;

        void setLabelPositions(const MapMeshResult & mapMeshResult);

        explicit InferenceLatentOrderDistributionResult(const PlayerAtTick & playerAtTick,
                                                        QueryPlayerAtTick & queryPlayerAtTick,
                                                        const orders::OrdersResult & ordersResult,
                                                        const MapMeshResult & mapMeshResult,
                                                        const InferenceLatentOrderResult & inferenceLatentOrderResult) :
            playerAtTick(playerAtTick),
            queryPlayerAtTick(queryPlayerAtTick),
            ordersResult(ordersResult),
            inferenceLatentOrderResult(inferenceLatentOrderResult) {
            setLabelPositions(mapMeshResult);
            variableLength = false;
            startTickColumn = 0;
            ticksPerEvent = 1;
            perTickPosLabels = true;
            for (size_t i = 0; i < orderLabelPositions.size(); i++) {
                posLabelsPositions.push_back(orderLabelPositions[i].toCSV("_"));
            }
        };

        vector<int64_t> filterByForeignKey(int64_t otherTableIndex) override {
            return queryPlayerAtTick.filterByForeignKey(otherTableIndex);
        }

        void oneLineToCSV(int64_t index, std::ostream &s) override {
            s << index << "," << playerAtTick.tickId[index] << ",";

            s << std::setprecision(2);
            bool first = true;
            for (const auto prob : inferenceLatentOrderResult.playerOrderProb[index]) {
                if (!first) {
                    s << ";";
                }
                s << prob;
                first = false;
            }
            s << std::endl;
        }

        [[nodiscard]]
        vector<string> getForeignKeyNames() override {
            return {"tick id"};
        }

        [[nodiscard]]
        vector<string> getOtherColumnNames() override {
            return {"order prob"};
        }
    };
}

#endif //CSKNOW_INFERENCE_LATENT_ORDER_DISTRIBUTION_H
