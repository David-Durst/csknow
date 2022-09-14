# CSKnow

This project is [my](https://davidbdurst.com/) research on understanding player behavior in video games. The key components of the project are

## `analytics`

The C++ code for implementing the bots, the analytics database querying demo files, and the server sending these queries to the web visualization.

## `demo_generator`

The docker config files and configurations for running CSGO servers in docker containers, thus generating demos to analyze from bots

## `demo_parser`

The golang demo parser. This parser uploads CSV results to AWS S3 by default.

## `download_s3_csvs` 

Download parsed demos' CSVs to the `local_data` folder. The `analytics` code reads the CSVs out of the `local_data` folder.

## `web_vis`

The front-end code for visualizing queries computed by the `analytics` database.


