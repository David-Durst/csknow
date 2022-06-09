#include "navmesh/nav_file.h"
#include <iostream>
#include <memory>
#include <limits>
#include <cmath>
#include <csignal>
#define PLAYER_WIDTH 32

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
        m_area_ids_to_indices.clear( );

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

    std::optional< std::vector< vec3_t > > nav_file::find_path( vec3_t from, vec3_t to ) {
        auto start = reinterpret_cast< void* >( get_nearest_area_by_position( from ).get_id( ) );
        auto end = reinterpret_cast< void* >( get_nearest_area_by_position( to ).get_id( ) );
        std::vector< vec3_t > path = { };
        if (start == end) {
            path.push_back( to );
            return path;
        }

        float total_cost = 0.f;
        micropather::MPVector< void* > path_area_ids = { };

        if ( m_pather->Solve( start, end, &path_area_ids, &total_cost ) != 0 ) {
            return {};
        }

        for ( std::size_t i = 0; i < path_area_ids.size( ); i++ ) {
            nav_area& area = m_areas[m_area_ids_to_indices[LO_32( path_area_ids[ i ] )]];
            // smooth paths by adding intersections between nav areas after the first 
            // chose area in between nearest location in area i from center of i-1
            // and nearest location in area i-1 from center of i
            // as this will have max distance on either side for player to fit through
            if ( i != 0 ) {
                nav_area& last_area = m_areas[m_area_ids_to_indices[LO_32( path_area_ids[ i - 1 ] )]];
                // nw is min values, se is max value, so checking if x or y is the meeting point
                bool last_area_x_lesser = area.m_nw_corner.x == last_area.m_se_corner.x;
                bool area_x_lesser = area.m_se_corner.x == last_area.m_nw_corner.x;
                bool last_area_y_lesser = area.m_nw_corner.y == last_area.m_se_corner.y;
                bool x_lesser = last_area_x_lesser || area_x_lesser;
                vec3_t middle;
                if (x_lesser) {
                    middle.x = last_area_x_lesser ? area.m_nw_corner.x : last_area.m_nw_corner.x;
                    float max_valid_y = std::min(area.m_se_corner.y, last_area.m_se_corner.y);
                    float min_valid_y = std::min(area.m_nw_corner.y, last_area.m_nw_corner.y);
                    middle.y = (max_valid_y + min_valid_y) / 2.f;
                }
                else {
                    middle.y = last_area_y_lesser ? area.m_nw_corner.y : last_area.m_nw_corner.y;
                    float max_valid_x = std::min(area.m_se_corner.x, last_area.m_se_corner.x);
                    float min_valid_x = std::min(area.m_nw_corner.x, last_area.m_nw_corner.x);
                    middle.x = (max_valid_x + min_valid_x) / 2.f;

                }
                middle.z = (area.get_center().z + last_area.get_center().z) / 2.f;
                path.push_back( middle );
            }
            path.push_back( area.get_center( ) );
        }

        path.push_back( to );

        return path;
    }

    std::optional< std::vector< PathNode > > nav_file::find_path_detailed( vec3_t from, vec3_t to ) {
        const nav_area & fromArea = get_nearest_area_by_position( from );
        const nav_area & toArea = get_nearest_area_by_position( to );
        auto start = reinterpret_cast< void* >( fromArea.get_id() );
        auto end = reinterpret_cast< void* >( toArea.get_id( ) );
        std::vector< PathNode > path = { };
        if (start == end) {
            path.push_back( { false, get_nearest_area_by_position( to ).get_id( ), 0, to } );
            return path;
        }

        float total_cost = 0.f;
        micropather::MPVector< void* > path_area_ids = { };

        if ( m_pather->Solve( start, end, &path_area_ids, &total_cost ) != 0 ) {
            return {};
        }

        for ( std::size_t i = 0; i < path_area_ids.size( ); i++ ) {
            nav_area& area = m_areas[m_area_ids_to_indices[LO_32( path_area_ids[ i ] )]];
            // smooth paths by adding intersections between nav areas after the first
            // chose area in between nearest location in area i from center of i-1
            // and nearest location in area i-1 from center of i
            // as this will have max distance on either side for player to fit through
            if ( i != 0 ) {
                nav_area& last_area = m_areas[m_area_ids_to_indices[LO_32( path_area_ids[ i - 1 ] )]];
                // nw is min values, se is max value, so checking if x or y is the meeting point
                bool last_area_x_lesser = last_area.get_max_corner().x <= area.get_min_corner().x;
                bool area_x_lesser = area.get_max_corner().x <= last_area.get_min_corner().x;
                bool last_area_y_lesser = last_area.get_max_corner().y <= area.get_min_corner().y;
                bool area_y_lesser = area.get_max_corner().y <= last_area.get_min_corner().y;
                if (!last_area_x_lesser && !area_x_lesser && !last_area_y_lesser && !area_y_lesser) {
                    std::cout << "bad path from area " << last_area.get_id() << " to " << area.get_id() << std::endl;
                }
                bool x_lesser = last_area_x_lesser || area_x_lesser;
                vec3_t middle;
                if (x_lesser) {
                    middle.x = last_area_x_lesser ? area.m_nw_corner.x : last_area.m_nw_corner.x;
                    float max_valid_y = std::min(area.m_se_corner.y, last_area.m_se_corner.y);
                    float min_valid_y = std::max(area.m_nw_corner.y, last_area.m_nw_corner.y);
                    middle.y = (max_valid_y + min_valid_y) / 2.f;
                }
                else {
                    middle.y = last_area_y_lesser ? area.m_nw_corner.y : last_area.m_nw_corner.y;
                    float max_valid_x = std::min(area.m_se_corner.x, last_area.m_se_corner.x);
                    float min_valid_x = std::max(area.m_nw_corner.x, last_area.m_nw_corner.x);
                    middle.x = (max_valid_x + min_valid_x) / 2.f;

                }
                // if falling off cliff, never able to hit half way between top and bottom
                // same logic if jumping
                // take bottom z
                middle.z = last_area.get_center().z;
                path.push_back( { true, last_area.get_id(), area.get_id(), middle } );
            }
            path.push_back( { false, area.get_id(), 0, area.get_center( ) } );
        }

        path.push_back( { false, get_nearest_area_by_position( to ).get_id( ), 0, to } );

        return path;
    }

    const nav_area& nav_file::get_area_by_id( std::uint32_t id ) const {
        for ( auto& area : m_areas ) {
            if ( area.get_id( ) == id )
                return area;
        }

        throw std::runtime_error( "nav_file::get_area_by_id: failed to find area" );
    }

    const nav_area& nav_file::get_area_by_id_fast( std::uint32_t id ) const {
        return m_areas[m_area_ids_to_indices.find(id)->second];
    }

    std::string nav_file::get_place(std::uint16_t id) const {
        if (id < m_places.size()) {
            std::string result = m_places[id];
            result.erase(result.find('\0'));
            return result;
        }
        else {
            return "INVALID";
        }
    }

    nav_area& nav_file::get_area_by_position( vec3_t position ) {
        for ( auto& area : m_areas ) {
            if ( area.is_within( position ) )
                return area;
        }

        throw std::runtime_error( "nav_file::get_area_by_position: failed to find area" );
    }

    // https://stackoverflow.com/questions/5254838/calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
    float nav_file::get_point_to_area_distance( vec3_t position, const nav_area& area) const {
        float dx = std::max(area.m_nw_corner.x - position.x,
                std::max(0.f, position.x - area.m_se_corner.x));
        float dy = std::max(area.m_nw_corner.y - position.y, 
                std::max(0.f, position.y - area.m_se_corner.y));
        float dz = std::max(area.get_min_corner().z - position.z,
                std::max(0.f, position.z - area.get_max_corner().z));
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    float nav_file::get_point_to_area_distance_2d( vec3_t position, const nav_area& area) const {
        float dx = std::max(area.m_nw_corner.x - position.x,
                            std::max(0.f, position.x - area.m_se_corner.x));
        float dy = std::max(area.m_nw_corner.y - position.y,
                            std::max(0.f, position.y - area.m_se_corner.y));
        return std::sqrt(dx * dx + dy * dy);
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
        if (position.z < area.get_min_corner().z) {
            result.z = area.m_nw_corner.z;
        }
        else if (position.z > area.get_max_corner().z) {
            result.z = area.m_se_corner.z;
        }
        return result;
    }

    const nav_area& nav_file::get_nearest_area_by_position( vec3_t position ) const {
        float nearest_area_distance = std::numeric_limits<float>::max();        
        size_t nearest_area_id = -1;

        for ( size_t area_id = 0; area_id < m_areas.size(); area_id++) {
            const nav_area& area = m_areas[area_id];
            // skip bugged areas with no connections
            if ( area.m_connections.empty() ) {
                continue;
            }
            if ( area.is_within_3d( position ) ) {
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

    void nav_file::remove_incoming_edges_to_areas( std::set<std::uint32_t> ids ) {
        for ( size_t area_id = 0; area_id < m_areas.size(); area_id++) {
            std::vector< nav_connect_t >& area_connections = m_areas[area_id].m_connections;
            area_connections.erase(std::remove_if(
                    area_connections.begin(),
                    area_connections.end(),
                    [&](nav_connect_t con) { return ids.find(con.id) != ids.end(); }),
                area_connections.end());
        }
    }
}
