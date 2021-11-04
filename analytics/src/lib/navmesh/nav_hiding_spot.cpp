#include "navmesh/nav_hiding_spot.h"

namespace nav_mesh {
	nav_hiding_spot::nav_hiding_spot( nav_buffer& buffer ) {
		load( buffer );
	}

	void nav_hiding_spot::load( nav_buffer& buffer ) {
		m_id = buffer.read< std::uint32_t >( );
		buffer.read( &m_pos, sizeof(vec3_t) );
		m_flags = buffer.read< std::uint8_t >( );
	}
}