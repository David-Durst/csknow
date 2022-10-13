//
// Created by durst on 10/11/22.
//

#ifndef CSKNOW_BITSET_UINT8_H
#define CSKNOW_BITSET_UINT8_H
#include <vector>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <omp.h>
#include <stdexcept>
using std::vector;
using std::size_t;

namespace csknow {
    template<size_t N>
    class Bitset {
        vector<uint8_t> data;
    public:
        inline size_t internalLength() { return static_cast<size_t>(std::ceil(N / 8.)); }
        Bitset() : data(internalLength(), 0) { }
        Bitset(vector<uint8_t> && data) : data(std::move(data)) {
            if (data.size() != internalLength()) {
                throw std::runtime_error("bitset created with data of wrong length");
            }
        }
        void operator=(vector<uint8_t> && newData) {
            data = std::move(newData);
            if (data.size() != internalLength()) {
                throw std::runtime_error("bitset assigned new data of wrong length");
            }
        }

        bool operator[](size_t i) const { return ((data[i / 8] >> (i%8)) & 1) != 0; };
        void set(size_t i, bool v) {
            if (v) {
                data[i / 8] |= 1 << (i%8);
            }
            else {
                uint8_t mask = 255;
                mask ^= 1 << (i%8);
                data[i / 8] &= mask;
            }
        };


        Bitset & operator|=(const Bitset<N> & other) {
#pragma omp for simd
            for (size_t i = 0; i < data.size(); i++) {
                data[i] |= other.data[i];
            }
            return *this;
        }

        Bitset & operator&=(const Bitset<N> & other) {
#pragma omp for simd
            for (size_t i = 0; i < data.size(); i++) {
                data[i] &= other.data[i];
            }
            return *this;
        }

        bool any() {
            for (size_t i = 0; i < data.size(); i++) {
                if (data[i] != 0) {
                    return true;
                }
            }
            return false;
        }

        bool all() {
            for (size_t i = 0; i < data.size(); i++) {
                if (data[i] != 255) {
                    return false;
                }
            }
            return true;
        }

        void flip() {
#pragma omp for simd
            for (size_t i = 0; i < data.size(); i++) {
                data[i] = ~data[i];
            }
        }

        void reset() {
            std::fill(data.begin(), data.end(), 0);
        }

        size_t size() { return N; }

        void assignSlice(const vector<uint8_t> & source, size_t startByte) {
            data.clear();
            std::copy(source.begin() + startByte, source.begin() + startByte + data.size(),
                      std::back_inserter(data));
        }

        const vector<uint8_t> & getInternal() const { return data; }
    };
}

#endif //CSKNOW_BITSET_UINT8_H
