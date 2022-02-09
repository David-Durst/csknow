#include "navmesh/nav_file.h"
#include <iostream>
#include <memory>
#include <limits>
#include <cmath>

namespace nav_mesh {
    nav_file::nav_file( std::string_view nav_mesh_file ) {
        
        load( nav_mesh_file );
    }

    void nav_file::load( std::string_view nav_mesh_file ) {
        if ( !m_pather )
            m_pather = std::make_unique< micropather::MicroPather >( this );

        m_pather->Reset( );
        m_areas.clear( );
        m_places.clear( );

        m_buffer.load_from_file( nav_mesh_file );

        if ( m_buffer.read< std::uint32_t >( ) != m_magic )
            throw std::runtime_error( "nav_file::load: magic mismatch" );

        m_version = m_buffer.read< std::uint32_t >( );

        if ( m_version != 16 )
            throw std::runtime_error( "nav_file::load: version mismatch" );

        m_sub_version = m_buffer.read< std::uint32_t >( );
        m_source_bsp_size = m_buffer.read< std::uint32_t >( );
        m_is_analyzed = m_buffer.read< std::uint8_t >( );        
        m_place_count = m_buffer.read< std::uint16_t >( );
        
        for ( std::uint16_t i = 0; i < m_place_count; i++ ) {
            auto place_name_length = m_buffer.read< std::uint16_t >( );
            std::string place_name( place_name_length, 0 );

            m_buffer.read( place_name.data( ), place_name_length );
            m_places.push_back( place_name );
        }

        m_has_unnamed_areas = m_buffer.read< std::uint8_t >( ) != 0;
        m_area_count = m_buffer.read< std::uint32_t >( );

        if ( m_area_count == 0 )
            throw std::runtime_error( "nav_file::load: no areas" );

        for ( std::uint32_t i = 0; i < m_area_count; i++ ) {
            nav_area area( m_buffer );
            m_areas.push_back( area );
        }

        m_buffer.clear( );

        for ( size_t area_id = 0; area_id < m_areas.size(); area_id++ ) {
            m_area_ids_to_indices.insert({m_areas[area_id].get_id(), area_id});
        }

    }

    std::vector< vec3_t > nav_file::find_path( vec3_t from, vec3_t to ) {
        auto start = reinterpret_cast< void* >( get_nearest_area_by_position( from ).get_id( ) );
        auto end = reinterpret_cast< void* >( get_nearest_area_by_position( to ).get_id( ) );

        float total_cost = 0.f;
        micropather::MPVector< void* > path_area_ids = { };

        if ( m_pather->Solve( start, end, &path_area_ids, &total_cost ) != 0 )
            throw std::runtime_error( "nav_file::find_path: couldn't find path" );

        std::vector< vec3_t > path = { };
        for ( std::size_t i = 0; i < path_area_ids.size( ); i++ ) {
            nav_area& area = m_areas[m_area_ids_to_indices[LO_32( path_area_ids[ i ] )]];
            // smooth paths by adding intersections between nav areas after the first 
            // chose area in between nearest location in area i from center of i-1
            // and nearest location in area i-1 from center of i
            // as this will have max distance on either side for player to fit through
            if ( i != 0 ) {
                nav_area& last_area = m_areas[m_area_ids_to_indices[LO_32( path_area_ids[ i - 1 ] )]];
                vec3_t nearest_0 = get_nearest_point_in_area(last_area.get_center(), area);
                vec3_t nearest_1 = get_nearest_point_in_area(area.get_center(), last_area);
                vec3_t middle = {
                    (nearest_0.x + nearest_1.x) / 2.f,
                    (nearest_0.y + nearest_1.y) / 2.f,
                    (nearest_0.z + nearest_1.z) / 2.f
                };
                path.push_back( middle );
            }
            path.push_back( area.get_center( ) );
        }

        path.push_back( to );

        return path;
    }

    nav_area& nav_file::get_area_by_id( std::uint32_t id ) {
        for ( auto& area : m_areas ) {
            if ( area.get_id( ) == id )
                return area;
        }

        throw std::runtime_error( "nav_file::get_area_by_id: failed to find area" );
    }

    nav_area& nav_file::get_area_by_position( vec3_t position ) {
        for ( auto& area : m_areas ) {
            if ( area.is_within( position ) )
                return area;
        }

        throw std::runtime_error( "nav_file::get_area_by_position: failed to find area" );
    }

    // https://stackoverflow.com/questions/5254838/calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
    float nav_file::get_point_to_area_distance( vec3_t position, nav_area& area) {
        float dx = std::max(area.m_nw_corner.x - position.x,
                std::max(0.f, position.x - area.m_se_corner.x));
        float dy = std::max(area.m_nw_corner.y - position.y, 
                std::max(0.f, position.y - area.m_se_corner.y));
        float dz = std::max(area.m_nw_corner.z - position.z,
                std::max(0.f, position.z - area.m_se_corner.z));
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    vec3_t nav_file::get_nearest_point_in_area( vec3_t position, nav_area& area) {
        vec3_t result = position;
        if (position.x < area.m_nw_corner.x) {
            result.x = area.m_nw_corner.x;
        }
        else if (position.x > area.m_se_corner.x) {
            result.x = area.m_se_corner.x;
        }
        if (position.y < area.m_nw_corner.y) {
            result.y = area.m_nw_corner.y;
        }
        else if (position.y > area.m_se_corner.y) {
            result.y = area.m_se_corner.y;
        }
        if (position.z < area.m_nw_corner.z) {
            result.z = area.m_nw_corner.z;
        }
        else if (position.z > area.m_se_corner.z) {
            result.z = area.m_se_corner.z;
        }
        return result;
    }

    nav_area& nav_file::get_nearest_area_by_position( vec3_t position ) {
        float nearest_area_distance = std::numeric_limits<float>::max();        
        size_t nearest_area_id = -1;

        for ( size_t area_id = 0; area_id < m_areas.size(); area_id++) {
            nav_area& area = m_areas[area_id];
            if ( area.is_within( position ) ) {
                return area;
            }
            float other_distance = get_point_to_area_distance( position, area);
            if ( other_distance < nearest_area_distance ) {
                nearest_area_distance = other_distance;
                nearest_area_id = area_id;
            }
        }

        if ( nearest_area_id == -1) {
            throw std::runtime_error( "nav_file::get_nearest_area_by_position: no areas" );
        }
        else {
            return m_areas[nearest_area_id];
        }

    }
}
