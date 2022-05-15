#include "navmesh/nav_area.h"
#include <algorithm>
#include <iostream>

namespace nav_mesh {
	nav_area::nav_area( nav_buffer& buffer ) {
		load( buffer );
	}

	bool nav_area::is_within( vec3_t position ) const {
		if ( position.x < m_nw_corner.x )
			return false;

		if ( position.x > m_se_corner.x )
			return false;

		if ( position.y < m_nw_corner.y )
			return false;

		if ( position.y > m_se_corner.y )
			return false;

		return true;
	}

    bool nav_area::is_within_3d( vec3_t position, float z_tolerance ) const {
        if ( position.x < m_nw_corner.x )
            return false;

        if ( position.x > m_se_corner.x )
            return false;

        if ( position.y < m_nw_corner.y )
            return false;

        if ( position.y > m_se_corner.y )
            return false;

        if ( position.z < m_nw_corner.z - z_tolerance )
            return false;

        if ( position.z > m_nw_corner.z + z_tolerance )
            return false;

        return true;
    }

	void nav_area::load( nav_buffer& buffer ) {
		m_id = buffer.read< std::uint32_t >( );
		m_attribute_flags = buffer.read< std::uint32_t >( );

		buffer.read( &m_nw_corner, sizeof(vec3_t) );
		buffer.read( &m_se_corner, sizeof(vec3_t) );

		m_center = ( m_nw_corner + m_se_corner ) * .5f;

		auto corner_delta = m_se_corner - m_nw_corner;
		if ( corner_delta.x > 0.f && corner_delta.y > 0.f ) {
			m_inv_dx_corners = 1.f / corner_delta.x;
			m_inv_dy_corners = 1.f / corner_delta.y;
		} else
			m_inv_dx_corners = m_inv_dy_corners = 0.f;

		m_ne_z = buffer.read< float >( );
		m_sw_z = buffer.read< float >( );

		for ( std::uint32_t i = 0; i < 4; i++ ) {
			auto connection_count = buffer.read< std::uint32_t >( );

			for ( std::uint32_t j = 0; j < connection_count; j++ ) {
				auto connect_id = buffer.read< std::uint32_t >( );

				if ( connect_id == m_id )
					continue;

				m_connections.push_back( nav_connect_t( connect_id ) );
			}
		}

		auto hiding_spot_count = buffer.read< std::uint8_t >( );
		for ( std::uint32_t i = 0; i < hiding_spot_count; i++ )
			m_hiding_spots.push_back( nav_hiding_spot( buffer ) );

		auto encounter_path_count = buffer.read< std::uint32_t >( );
		for ( std::uint32_t i = 0; i < encounter_path_count; i++ ) {
			nav_spot_encounter_t spot_encounter = { };

			spot_encounter.from.id = buffer.read< std::uint32_t >( );
			spot_encounter.from_direction = buffer.read< std::uint8_t >( );

			spot_encounter.to.id = buffer.read< std::uint32_t >( );
			spot_encounter.to_direction = buffer.read< std::uint8_t >( );

			auto spot_count = buffer.read< std::uint8_t >( );
			
			nav_spot_order_t spot_order = { };
			for ( std::uint8_t i = 0; i < spot_count; i++ ) {
				spot_order.id = buffer.read< std::uint32_t >( );
				spot_order.t = float( buffer.read< std::uint8_t >( ) ) / 255.f;
				spot_encounter.spot_order.push_back( spot_order );
			}

			m_spot_encounters.push_back( spot_encounter );
		}

		m_place = buffer.read< std::uint16_t >( ) - 1;

		for ( std::uint32_t i = 0; i < 2; i++ ) {
			auto ladder_count = buffer.read< std::uint32_t >( );

			for ( std::uint32_t j = 0; j < ladder_count; j++ ) {
				nav_ladder_connect_t ladder_connect( buffer.read< std::uint32_t >( ) );

				bool skip = false;
				for ( std::uint32_t j = 0; j < m_ladder_connections[ i ].size( ); j++ ) {
					if ( m_ladder_connections[ i ][ j ].id == ladder_connect.id ) {
						skip = true;
						break;
					}
				}

				if ( skip )
					continue;

				m_ladder_connections[ i ].push_back( ladder_connect );
			}
		}

		for ( std::uint32_t i = 0; i < 2; i++ )
			m_earliest_occupy_time[ i ] = buffer.read< float >( );

		for ( std::uint32_t i = 0; i < 4; i++ )
			m_light_intensity[ i ] = buffer.read< float >( );

		auto visible_area_count = buffer.read< std::uint32_t >( );
		for ( std::uint32_t i = 0; i < visible_area_count; i++ ) {
			nav_area_bind_info_t area_bind_info = { };

			area_bind_info.id = buffer.read< std::uint32_t >( );
			area_bind_info.attributes = buffer.read< std::uint8_t >( );

			m_potentially_visible_areas.push_back( area_bind_info );
		}

		m_inherit_visibility_from.id = buffer.read< std::uint32_t >( );

		//Credits: https://github.com/mrazza/gonav/blob/master/parser.go#L258-L260
		auto unknown_count = buffer.read< std::uint8_t >( );
		for ( std::uint8_t i = 0; i < unknown_count; i++ )
			buffer.skip( 0xE );
	}
}