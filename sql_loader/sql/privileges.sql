CREATE USER readonly WITH PASSWORD 'readonly';
USE csknow;
GRANT CONNECT ON DATABASE csknow TO readonly;
CREATE USER 'readonly'@'%' IDENTIFIED BY 'readonly';
GRANT SELECT ON *.* TO 'readonly'@'%';
FLUSH PRIVILEGES;
CREATE DATABASE IF NOT EXISTS csknow
