#pragma once
#define LO_32(x) (*( (uint32_t *) &x))

namespace nav_mesh {
	class vec3_t {
	public:
		vec3_t operator+( const vec3_t other )	{ return { x + other.x, y + other.y, z + other.z }; }
		vec3_t operator-( const vec3_t other )	{ return { x - other.x, y - other.y, z - other.z }; }
		vec3_t operator*( const float m )		{ return { x * m, y * m, z* m }; }

		float x = 0.f, y = 0.f, z = 0.f;
	};

	struct nav_area_bind_info_t {
		union {
			std::uint32_t id;
			void* area = nullptr;
		};

		std::uint8_t attributes = 0;
	};

	class nav_area_critical_data {
	protected:
		vec3_t m_nw_corner = { },
			m_se_corner = { },
			m_center = { };

		float m_inv_dx_corners = 0.f,
			m_inv_dy_corners = 0.f,
			m_ne_z = 0.f,
			m_sw_z = 0.f;
	};

	struct nav_connect_t {
		nav_connect_t( ) { }
		nav_connect_t( std::uint32_t connect_id ) {
			id = connect_id;
		}

		union {
			std::uint32_t id;
			void* area = nullptr;
		};
	};

	struct nav_ladder_connect_t {
		nav_ladder_connect_t( ) { }
		nav_ladder_connect_t( std::uint32_t connect_id ) {
			id = connect_id;
		}

		union {
			std::uint32_t id;
			void* ladder = nullptr;
		};
	};

	struct nav_spot_order_t {
		float t = 0.f;
		
		union {
			std::uint32_t id;
			void* hiding_spot = nullptr;
		};
	};

	struct nav_spot_encounter_t {	
		nav_connect_t from = { }, to = { };		
		std::uint8_t from_direction = 0, to_direction = 0;
		
		std::vector< nav_spot_order_t > spot_order = { };
	};
}