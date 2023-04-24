//
// Created by durst on 12/25/22.
//

#ifndef CSKNOW_STREAMING_ENGAGEMENT_AIM_H
#define CSKNOW_STREAMING_ENGAGEMENT_AIM_H

#include "queries/training_moments/training_engagement_aim.h"
#include "bots/streaming_bot_database.h"
#include "bots/streaming_moments/streaming_fire_history.h"
#include "queries/inference_moments/inference_engagement_aim.h"
#include <torch/script.h>
#include <filesystem>
#include <ATen/Parallel.h>
#include <random>

namespace fs = std::filesystem;

namespace csknow::engagement_aim {
    AimWeaponType weaponIdToWeaponType(int32_t weaponId);

    struct EngagementAimTarget {
        // if set to invalid, then need to make a target from a position
        CSGOId csgoId;
        Vec3 pos;

        bool isPlayer() const {
            return csgoId != INVALID_ID;
        }

        bool operator==(const EngagementAimTarget & other) const {
            if (isPlayer() && other.isPlayer()) {
                return csgoId == other.csgoId;
            }
            else if (!isPlayer() && !other.isPlayer()) {
                return pos == other.pos;
            }
            else {
                return false;
            }
        }

        bool operator!=(const EngagementAimTarget & other) const {
            return !this->operator==(other);
        }
    };

    typedef unordered_map<CSGOId, EngagementAimTarget> ClientTargetMap;

    class StreamingEngagementAim {
        EngagementAimTickData computeOneTickData(StreamingBotDatabase & db,
                                                 const fire_history::StreamingFireHistory & streamingFireHistory,
                                                 CSGOId attackerId, const EngagementAimTarget & target,
                                                 size_t attackerStateOffset, size_t victimStateOffset,
                                                 bool firstEngagementTick);
        void predictNewAngles(const StreamingBotDatabase & db);

        torch::jit::script::Module module;
        bool resetInternal = false;
    public:
        int printAimTicks = 0;
        std::fstream aimTicksFile;

        std::random_device rd;
        std::mt19937 gen;
        std::normal_distribution<> outXYDist{0.,0.1};
        std::uniform_real_distribution<> fakeAttackDist{0, 1.};

        StreamingEngagementAim(const string & navPath) :
            aimTicksFile(fs::path(navPath) / fs::path("..") / fs::path("..") /
            fs::path("aim_ticks.csv"), std::fstream::out), gen(rd()) {

            fs::path modelPath = fs::path(navPath) / fs::path("..") /
                fs::path("..") / fs::path("learn_bot") / fs::path("models") /
                fs::path("engagement_aim_model") / fs::path("script_model.pt");

            try {
                //auto tmp_module = torch::jit::load(modelPath);
                //module = optimize_for_inference(tmp_module);
            }
            catch (const c10::Error& e) {
                std::cerr << "error loading engagement aim model\n" << e.msg() << std::endl;
            }
        }

        StreamingClientHistory<EngagementAimTickData> engagementAimPlayerHistory;

        ClientTargetMap currentClientTargetMap, priorClientTargetMap;
        unordered_map<CSGOId, StreamingPinId> playerToVictimLastAlivePos;
        unordered_map<CSGOId, Vec2> playerToVictimEngagementFirstIdealViewAngle;
        unordered_map<CSGOId, bool> playerToVictimFirstVisibleFrame;
        unordered_map<CSGOId, Vec2> playerToNewAngle;
        unordered_map<CSGOId, Vec2> playerToDeltaAngle;
        unordered_map<CSGOId, bool> playerToFiring;
        unordered_map<CSGOId, uint32_t> playerToManualOverride;

        void reset() {
            resetInternal = true;
            playerToNewAngle.clear();
            playerToDeltaAngle.clear();
        }

        void addTickData(StreamingBotDatabase & db,
                         const fire_history::StreamingFireHistory & streamingFireHistory);
    };
}

#endif //CSKNOW_STREAMING_ENGAGEMENT_AIM_H
