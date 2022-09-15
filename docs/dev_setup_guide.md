# Developer Setup Guide
These are instructions for jetting up development on CSGO's data analytics and bot systems.
If you just want to play against the bots, see my
[user instructions](https://github.com/David-Durst/csknow/blob/master/docs/bot_user_instructions.md).

## Prerequisites
These instructions assume Ubuntu 22.04 OS with sufficient space to install a
CSGO server (~30 GB). If your personal computer doesn't match these
specifications, I recommend creating a machine on AWS using [my
guide](https://github.com/David-Durst/csknow/blob/master/docs/aws_machine_creation_guide.md).

## Data Analytics Instructions
1. Open a terminal in the root CSKnow directory
1. Install the C++ compiler, C++ build system, and openssl dependency. Run
   `analytics/scripts/install_dependencies.sh` to install cmake/gcc/openssl. Right now, I
   use gcc-11/g++-11, but I presume clang or future version of gcc should work
   fine. I just require C++-17 features.
   1. **Note:** run this script as your normal user, it will request sudo access when necessary
2. Install golang. I recommend doing this by installing the
   [goenv](https://github.com/Spacewalkio/Goenv) Go version manager. Run
   `demo_parser/scripts/install_dependencies.sh` to install goenv to `/opt`,
   Go 1.19 to a location in your home directory, and jq via the package manager
   (jq used to get latest goenv version).
    3. **Note:** Running this script with the y flag
       (`demo_parser/scripts/install_dependencies.sh y`) will add goenv and
       go to your path. Otherwise, it will print instructions for you to add
       them to the path in `~/.profile`.
    4. However you choose to add goenv and go to your path, make sure to log out
       and relogin if you add to your path using `~/.profile` .
3. **For David Durst Only** (all others lack access to AWS): download demos from
   AWS, process them into CSVs for the C++ database, and upload the CSVs to aws
   by `cd` ing into `demo_parser` and running running
   `scripts/process_aws_on_local.sh`
    1. For others, if you download a demo into the `demo_parser` directory and
       run ` go run cmd/main.go -l -n demo_name.dem` the parser will run on that
       demo and write csvs to `demo_parser/csv_outputs` .
    2. This won't integrate with the rest of the data analytics pipeline, but it
       lets you experiment with the parser in isolation.
4. Download csvs from S3 that the C++ database will load (note: AWS
   permissions allow anyone to run this step):
    1. Install Docker using `./demo_generator/docker_setup.sh`
    1. `cd download_s3_csvs`
    2. run `./build.sh` to build the S3 downloader Docker image
    3. run `./start.sh` to download the csvs into `local_data`
5. Run the C++ database by `cd` ing into `analytics` and running `./scripts/server_run.sh`
6. Visualize the C++ database query results
    1. `cd` into `web_vis`
    2. Install nvm (the npm version manager) `./scripts/install_nvm.sh`
    2. Close and reopen your terminal for the installation to take effect.
    3. Install npm/node by running `./scripts/install_node.sh` 
    4. Install the package by running `npm install`
    5. Continuously rebuild the visualization system as code changes by running
       `npm run watch` (**note:** must run this at least once, even if not making
       any changes, can quit out when compilation finished)
    6. Open `index.html` in a web browser. I use firefox, but chrome should also work

## Bot Instructions
**TODO**
