-- Create a group
CREATE ROLE readaccess;

-- Grant access to existing tables
GRANT USAGE ON SCHEMA public TO readaccess;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readaccess;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO readaccess;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;

-- Grant access to future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO readaccess;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON SEQUENCES TO readaccess;

-- Create a final user with password
CREATE USER readonly WITH PASSWORD 'readonly';
GRANT readaccess TO readonly;
