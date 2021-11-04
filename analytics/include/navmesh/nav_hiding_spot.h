#pragma once
#include "nav_buffer.h"
#include "nav_structs.h"

namespace nav_mesh {
	class nav_hiding_spot {
	public:
		nav_hiding_spot( nav_buffer& buffer );

	private:
		void load( nav_buffer& buffer );

		std::uint8_t m_flags = 0;

		std::uint32_t m_id = 0;
		
		vec3_t m_pos = { };
	};
}