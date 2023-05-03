//
// Created by durst on 2/19/23.
//

#ifndef CSKNOW_ORDERS_H
#define CSKNOW_ORDERS_H
#include "queries/distance_to_places.h"
#include "bots/analysis/load_save_vis_points.h"
using std::string;
using std::vector;
using std::set;
using std::unordered_map;
using std::vector;

namespace csknow::orders {
    enum class OrderType {
        AOrder,
        BOrder
    };

    struct QueryOrder {
        std::vector<PlaceIndex> places;
        OrderType orderType;
    };

    struct OrderLandmarks {
        // start place is T space, end place is A or B site, other objective is other site
        PlaceIndex startPlace, endPlace, otherObjective;
    };

    class OrdersResult : public QueryResult {
        map<PlaceIndex, vector<PlaceIndex>> computeConnectedPlaces() const;
        std::set<PlaceIndex> computePlacesVisibleFromDestination(OrderLandmarks) const;
    public:
        std::vector<QueryOrder> orders;
        std::set<PlaceIndex> aPlacesVisibleFromDestination, bPlacesVisibleFromDestination;
        const VisPoints & visPoints;
        const MapMeshResult & mapMeshResult;
        const DistanceToPlacesResult & distanceToPlacesResult;
        explicit OrdersResult(const VisPoints & visPoints, const MapMeshResult & mapMeshResult,
                              const DistanceToPlacesResult & distanceToPlacesResult) :
            visPoints(visPoints), mapMeshResult(mapMeshResult), distanceToPlacesResult(distanceToPlacesResult) {
            variableLength = false;
            nonTemporal = true;
            overlay = true;
        }

        vector<int64_t> filterByForeignKey(int64_t) override {
            return {};
        }

        void oneLineToCSV(int64_t index, std::ostream &s) override {
            s << index << "," << (orders[index].orderType == OrderType::AOrder ? "A" : "B") << ",";
            for (size_t i = 0; i < orders[index].places.size(); i++) {
                if (i != 0) {
                    s << ";";
                }
                s << distanceToPlacesResult.places[orders[index].places[i]];
            }
            s << std::endl;
        }

        vector<string> getForeignKeyNames() override {
            return {"destination"};
        }

        vector<string> getOtherColumnNames() override {
            return {"place names"};
        }

        std::vector<QueryOrder> getOrdersByType(OrderType orderType) {
            std::vector<QueryOrder> result;
            for (const auto & queryOrder : orders) {
                if (queryOrder.orderType == orderType) {
                    result.push_back(queryOrder);
                }
            }
            return result;
        };

        void runQuery();
    };
}
#endif //CSKNOW_ORDERS_H
