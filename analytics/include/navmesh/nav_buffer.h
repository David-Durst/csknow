#pragma once
#include <vector>
#include <fstream>
#include <cstring>

namespace nav_mesh {
	class nav_buffer {
	public:
		~nav_buffer( );

		void load_from_file( std::string_view nav_mesh_file );

		void skip( std::size_t bytes_to_skip );

		void clear( );

		/*
		 *	The reason we don't erase the read bytes from the buffer is performance.
		 *	Benchmarks tested on cs_militia.nav:
		 *	Using std::vector::erase: average reading time of 7.5 seconds (!)
		 *	Adding read bytes to buffer pointer: 70ms (100x faster)
		 */
		template < typename T >
		T read( ) {
			T read = *reinterpret_cast< T* >( m_nav_buffer.data( ) + m_bytes_read );
			m_bytes_read += sizeof(T);

			return read;
		}

		void read( void* out_buffer, std::size_t bytes_to_read ) {
			memcpy( out_buffer, m_nav_buffer.data( ) + m_bytes_read, bytes_to_read );
			m_bytes_read += bytes_to_read;
		}

	private:
		std::size_t m_bytes_read = 0;

		std::vector< std::uint8_t > m_nav_buffer = { };
	};
}