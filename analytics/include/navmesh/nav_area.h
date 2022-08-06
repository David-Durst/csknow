#pragma once
#include "nav_hiding_spot.h"
#include "nav_structs.h"

enum class NavAttributeType : uint32_t
{
    NAV_MESH_INVALID		= 0,
    NAV_MESH_CROUCH			= 0x00000001,				// must crouch to use this node/area
    NAV_MESH_JUMP			= 0x00000002,				// must jump to traverse this area (only used during generation)
    NAV_MESH_PRECISE		= 0x00000004,				// do not adjust for obstacles, just move along area
    NAV_MESH_NO_JUMP		= 0x00000008,				// inhibit discontinuity jumping
    NAV_MESH_STOP			= 0x00000010,				// must stop when entering this area
    NAV_MESH_RUN			= 0x00000020,				// must run to traverse this area
    NAV_MESH_WALK			= 0x00000040,				// must walk to traverse this area
    NAV_MESH_AVOID			= 0x00000080,				// avoid this area unless alternatives are too dangerous
    NAV_MESH_TRANSIENT		= 0x00000100,				// area may become blocked, and should be periodically checked
    NAV_MESH_DONT_HIDE		= 0x00000200,				// area should not be considered for hiding spot generation
    NAV_MESH_STAND			= 0x00000400,				// bots hiding in this area should stand
    NAV_MESH_NO_HOSTAGES	= 0x00000800,				// hostages shouldn't use this area
    NAV_MESH_STAIRS			= 0x00001000,				// this area represents stairs, do not attempt to climb or jump them - just walk up
    NAV_MESH_NO_MERGE		= 0x00002000,				// don't merge this area with adjacent areas
    NAV_MESH_OBSTACLE_TOP	= 0x00004000,				// this nav area is the climb point on the tip of an obstacle
    NAV_MESH_CLIFF			= 0x00008000,				// this nav area is adjacent to a drop of at least CliffHeight

    NAV_MESH_FIRST_CUSTOM	= 0x00010000,				// apps may define custom app-specific bits starting with this value
    NAV_MESH_LAST_CUSTOM	= 0x04000000,				// apps must not define custom app-specific bits higher than with this value

    NAV_MESH_FUNC_COST		= 0x20000000,				// area has designer specified cost controlled by func_nav_cost entities
    NAV_MESH_HAS_ELEVATOR	= 0x40000000,				// area is in an elevator's path
    NAV_MESH_NAV_BLOCKER	= 0x80000000				// area is blocked by nav blocker ( Alas, needed to hijack a bit in the attributes to get within a cache line [7/24/2008 tom])
};


namespace nav_mesh {
	class nav_area : public nav_area_critical_data {
	public:
		nav_area( nav_buffer& buffer );

		vec3_t get_center( ) const								{ return m_center; }
        vec3_t get_min_corner( ) const								{ return {m_nw_corner.x, m_nw_corner.y, std::min(m_nw_corner.z, m_se_corner.z)}; }
        vec3_t get_max_corner( ) const								{ return {m_se_corner.x, m_se_corner.y, std::max(m_nw_corner.z, m_se_corner.z)}; }
		std::uint32_t get_id( )	const							{ return m_id; }
		
		const std::vector< nav_connect_t >& get_connections( ) const	{ return m_connections; }
		
		bool is_within( vec3_t position ) const;

        // max obstacle distance is 18 - https://developer.valvesoftware.com/wiki/Dimensions#Ground_Obstacle_Height
        // 1085->8964 requires large enough (cat mid ledge to cat) - 50 too big
        // 7574 ->7555 requires small enough (cat stairs) - 18 too small
        bool is_within_3d( vec3_t position, float z_tolerance = 30. ) const;

        void load( nav_buffer& buffer );

		std::uint16_t m_place = 0;

		std::uint32_t m_id = 0,
			m_attribute_flags = 0;

        bool is_flag_set(NavAttributeType attr) const {
            return (m_attribute_flags & static_cast<uint32_t>(attr)) > 0;
        }
	
		float m_light_intensity[ 4 ] = { 0.f };
		float m_earliest_occupy_time[ 2 ] = { 0.f };
		nav_area_bind_info_t m_inherit_visibility_from = { };

		std::vector< nav_connect_t > m_connections = { };		
		std::vector< nav_hiding_spot > m_hiding_spots = { };
		std::vector< nav_spot_encounter_t > m_spot_encounters = { };
		std::vector< nav_ladder_connect_t > m_ladder_connections[ 2 ] = { };
		std::vector< nav_area_bind_info_t > m_potentially_visible_areas = { };
	};
}