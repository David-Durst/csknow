#pragma once
#include "nav_area.h"
#include "micropather.h"
#include <math.h>
#include <memory>
#include <map>
#include <optional>
#include <set>
#include <algorithm>

namespace nav_mesh {
    struct PathNode {
        bool edgeMidpoint;
        uint32_t area1;
        uint32_t area2;
        vec3_t pos;
    };

	class nav_file : public micropather::Graph {
        std::set< uint32_t > m_areas_to_increase_cost;
	public:
		nav_file( ) { }
		nav_file( std::string_view nav_mesh_file );
		
		void load( std::string_view nav_mesh_file );

        std::optional< std::vector< vec3_t > > find_path( vec3_t from, vec3_t to );
        std::optional< std::vector< PathNode > > find_path_detailed( vec3_t from, vec3_t to );
        float compute_path_length( std::vector< PathNode > path);
        float compute_path_length_from_origin( vec3_t origin, std::vector< PathNode > path);

		//MicroPather implementation
		virtual float LeastCostEstimate( void* start, void* end ) {
			auto& start_area = get_area_by_id( LO_32( start ) );
			auto& end_area = get_area_by_id( LO_32( end ) );
			auto distance = start_area.get_center( ) - end_area.get_center( );

			return sqrtf( distance.x * distance.x + distance.y * distance.y + distance.z * distance.z );
		}

		virtual void AdjacentCost( void* state, micropather::MPVector< micropather::StateCost >* adjacent ) {
			auto& area = get_area_by_id( LO_32( state ) );
			auto& area_connections = area.get_connections( );

            float distance_adjustment = 0.f;

            if ( m_areas_to_increase_cost.find(area.get_id()) != m_areas_to_increase_cost.end() ) {
                for ( auto& connection : area_connections ) {
                    auto &connection_area = get_area_by_id(connection.id);
                    auto distance = connection_area.get_center() - area.get_center();
                    float distance_magnitude = sqrtf( distance.x * distance.x + distance.y * distance.y + distance.z * distance.z );
                    distance_adjustment = std::max(distance_magnitude, distance_adjustment);
                }
            }

			for ( auto& connection : area_connections ) {
				auto& connection_area = get_area_by_id( connection.id );
				auto distance = connection_area.get_center( ) - area.get_center( );

				micropather::StateCost cost = { reinterpret_cast< void* >( connection_area.get_id( ) ), 
					distance_adjustment * 10 + sqrtf( distance.x * distance.x + distance.y * distance.y + distance.z * distance.z ) };
				
				adjacent->push_back( cost );
			}
		}

		virtual void PrintStateInfo( void* state ) { }

		const nav_area& get_area_by_id( std::uint32_t id ) const;
		// added by durst since now have a lookup map but don't want to remove old implementaiton
        const nav_area& get_area_by_id_fast( std::uint32_t id ) const;
        // added by durst for maps that don't have places
        std::string get_place( std::uint16_t id ) const;
		nav_area& get_area_by_position( vec3_t position );
        float get_point_to_area_distance( vec3_t position, const nav_area& area) const;
        float get_point_to_area_distance_2d( vec3_t position, const nav_area& area) const;
        vec3_t get_nearest_point_in_area( vec3_t position, nav_area& area);
		const nav_area& get_nearest_area_by_position( vec3_t position ) const;
        void remove_incoming_edges_to_areas( std::set<std::uint32_t> ids );
        void build_connections_arrays( );
        std::set<std::uint32_t> get_sources_to_area( std::uint32_t id ) const;
        void set_areas_to_increase_cost( std::set<uint32_t> new_areas ) {
            m_areas_to_increase_cost = new_areas;
            m_pather->Reset();
        }

		std::unique_ptr< micropather::MicroPather > m_pather = nullptr;

		std::uint8_t m_is_analyzed = 0,
			m_has_unnamed_areas = 0;

		std::uint16_t m_place_count = 0; 

		std::uint32_t m_magic = 0xFEEDFACE, 
			m_version = 0,
			m_sub_version = 0,
			m_source_bsp_size = 0,
			m_area_count = 0;

		nav_buffer m_buffer = { };
		std::vector< nav_area > m_areas = { };
		std::vector< std::string > m_places = { };
        std::map< uint32_t, size_t > m_area_ids_to_indices;
        std::vector<size_t> connections; // store connections as a contiguous array of array indexes rather than area ids
        std::vector<size_t> connections_area_start, connections_area_length;
	};
}
