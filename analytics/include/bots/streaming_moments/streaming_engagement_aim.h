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

namespace fs = std::filesystem;

namespace csknow::engagement_aim {
    // https://tf2b.com/itemlist.php?gid=730
    enum class AimWeaponId {
        None = 0,
        Deagle,
        Dualies,
        FiveSeven,
        Glock,
        AK = 7,
        AUG,
        AWP,
        FAMAS,
        G3,
        Galil = 13,
        M249,
        M4A4 = 16,
        Mac10,
        P90 = 19,
        MP5 = 23,
        UMP,
        XM1014,
        Bizon,
        MAG7,
        Negev,
        SawedOff,
        Tec9,
        Zeus,
        P2000,
        MP7,
        MP9,
        Nova,
        P250,
        Scar = 38,
        SG553,
        SSG,
        M4A1S = 60,
        USPS,
        CZ = 63,
        R8
    };
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
    public:
        int printAimTicks = 0;
        StreamingEngagementAim(const string & navPath) {
            fs::path modelPath = fs::path(navPath) / fs::path("..") /
                fs::path("..") / fs::path("learn_bot") / fs::path("models") /
                fs::path("engagement_aim_model") / fs::path("script_model.pt");

            try {
                auto tmp_module = torch::jit::load(modelPath);
                module = optimize_for_inference(tmp_module);
            }
            catch (const c10::Error& e) {
                std::cerr << "error loading engagement aim model\n" << e.msg() << std::endl;
            }
        }

        StreamingClientHistory<EngagementAimTickData> engagementAimPlayerHistory;

        ClientTargetMap currentClientTargetMap, priorClientTargetMap;
        unordered_map<CSGOId, StreamingPinId> playerToVictimLastAlivePos;
        unordered_map<CSGOId, Vec3> playerToVictimEngagementFirstHeadPos;
        unordered_map<CSGOId, bool> playerToVictimFirstVisibleFrame;
        unordered_map<CSGOId, Vec2> playerToNewAngle;
        unordered_map<CSGOId, Vec2> playerToDeltaAngle;

        void addTickData(StreamingBotDatabase & db,
                         const fire_history::StreamingFireHistory & streamingFireHistory);
    };
}

#endif //CSKNOW_STREAMING_ENGAGEMENT_AIM_H
