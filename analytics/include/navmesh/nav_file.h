#pragma once
#include "nav_area.h"
#include "micropather.h"
#include <math.h>
#include <memory>
#include <map>

namespace nav_mesh {
	class nav_file : public micropather::Graph {
	public:
		nav_file( ) { }
		nav_file( std::string_view nav_mesh_file );
		
		void load( std::string_view nav_mesh_file );

		std::vector< vec3_t > find_path( vec3_t from, vec3_t to );

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

			for ( auto& connection : area_connections ) {
				auto& connection_area = get_area_by_id( connection.id );
				auto distance = connection_area.get_center( ) - area.get_center( );

				micropather::StateCost cost = { reinterpret_cast< void* >( connection_area.get_id( ) ), 
					sqrtf( distance.x * distance.x + distance.y * distance.y + distance.z * distance.z ) };
				
				adjacent->push_back( cost );
			}
		}

		virtual void PrintStateInfo( void* state ) { }

		nav_area& get_area_by_id( std::uint32_t id );
		nav_area& get_area_by_position( vec3_t position );
        float get_point_to_area_distance( vec3_t position, nav_area& area);
        vec3_t get_nearest_point_in_area( vec3_t position, nav_area& area);
		nav_area& get_nearest_area_by_position( vec3_t position );

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
	};
}
