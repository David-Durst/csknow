CREATE USER 'readonly'@'%' IDENTIFIED BY 'readonly';
GRANT SELECT ON *.* TO 'readonly'@'%';
GRANT CREATE TEMPORARY TABLES ON *.* TO 'readonly'@'%';
FLUSH PRIVILEGES;
CREATE DATABASE IF NOT EXISTS csknow
