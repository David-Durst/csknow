//
// Created by durst on 10/19/22.
//

#include "bots/analysis/save_map_state.h"
#include <png.h>

namespace csknow {

    MapState::MapState(const VisPoints & visPoints) : data{} {
        for (const auto & cellVisPoint : visPoints.getCellVisPoints()) {
            data[cellVisPoint.cellDiscreteCoordinates[0]][cellVisPoint.cellDiscreteCoordinates[1]] =
                std::numeric_limits<uint8_t>::max();
        }
    }

    void MapState::saveMapState(const fs::path &path) {
        // http://zarb.org/~gc/html/libpng.html
        /* create file */
        FILE *fp = fopen(path.c_str(), "wb");
        if (!fp) {
            throw std::runtime_error("[write_png_file] File could not be opened for writing: " + path.string());
        }

        //int x, y;

        //int width, height;
        //png_byte color_type;
        //png_byte bit_depth;

        //png_infop info_ptr;
        //int number_of_passes;
        //png_bytep * row_pointers;

        /* initialize stuff */
        png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

        if (!png_ptr) {
            throw std::runtime_error("[write_png_file] png_create_write_struct failed");
        }

        png_infop info_ptr = png_create_info_struct(png_ptr);
        if (!info_ptr) {
            throw std::runtime_error("[write_png_file] png_create_info_struct failed");
        }

        if (setjmp(png_jmpbuf(png_ptr))) {
            throw std::runtime_error("[write_png_file] Error during init_io");
        }

        png_init_io(png_ptr, fp);


        /* write header */
        if (setjmp(png_jmpbuf(png_ptr))) {
            throw std::runtime_error("[write_png_file] Error during writing header");
        }

        png_set_IHDR(png_ptr, info_ptr, NAV_CELLS_PER_ROW, NAV_CELLS_PER_ROW,
                     8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

        png_write_info(png_ptr, info_ptr);


        /* write bytes */
        if (setjmp(png_jmpbuf(png_ptr))) {
            throw std::runtime_error("[write_png_file] Error during writing bytes");
        }

        png_write_image(png_ptr, reinterpret_cast<png_bytep>(data[0].data()));


        /* end write */
        if (setjmp(png_jmpbuf(png_ptr))) {
            throw std::runtime_error("[write_png_file] Error during end of write");
        }

        png_write_end(png_ptr, NULL);

        /* cleanup heap allocation */
        for (y=0; y<height; y++)
            free(row_pointers[y]);
        free(row_pointers);

        fclose(fp);    }
}