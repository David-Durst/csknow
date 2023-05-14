//
// Created by durst on 5/14/23.
//
#include "bots/testing/state_checks.h"

string posConstraintOpToString(const PosConstraintOp & op) {
    switch (op) {
        case PosConstraintOp::LT:
            return "<";
        case PosConstraintOp::LTE:
            return "<=";
        case PosConstraintOp::GT:
            return ">";
        case PosConstraintOp::GTE:
            return ">=";
        case PosConstraintOp::EQ:
            return "==";
        default:
            throw std::runtime_error("invalid pos constraint in to string");
    }

}
