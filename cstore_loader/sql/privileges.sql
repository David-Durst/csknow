CREATE USER 'readonly'@'%' IDENTIFIED BY 'readonly';
GRANT SELECT ON *.* TO 'readonly'@'%';
FLUSH PRIVILEGES;
CREATE DATABASE IF NOT EXISTS csknow
