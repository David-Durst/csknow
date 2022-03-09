//
// Created by durst on 3/9/22.
//

#ifndef CSKNOW_ENUM_HELPERS_H
#define CSKNOW_ENUM_HELPERS_H

template <class T>
constexpr int enumAsInt(T enumElem) {
    return static_cast<std::underlying_type_t<T>>(enumElem);
}

template <class T>
constexpr T intAsEnum(int32_t intElem) {
    return static_cast<T>(intElem);
}

#endif //CSKNOW_ENUM_HELPERS_H
