# Bot User Instructions
These are instructions for using CSKnow bots: running a server that you can play
against. If you want to modify bot logic, please see
[my dev instructions](https://github.com/David-Durst/csknow/tree/docs/bot_dev_instructions.md)
for how to setup the complete development environment.

These instructions assume Ubuntu 22.04 OS with sufficient space to install a
CSGO server (~30 GB). If your personal computer doesn't match these
specifications, I recommend creating a machine on AWS using [my guide](https://github.com/David-Durst/csknow/tree/docs/aws_machine_creation_instructions.md).

## Warning
These instructions require creating a GSLT token. Make sure to follow these
instructions on a server you control. Otherwise, other people will be able to
use your Steam account through your GSLT token.

## Instructions
1. Checkout the CSKnow repo with Git
1. Open a terminal in the CSKnow `demo_generator` folder
2. Install docker: Run `docker_setup.sh`. 
   1. **Note:** run this script as your normal user, it will request sudo access when necessary
3. Build the docker image: Run `build.sh` 
4. Create the GSLT that registers the server under your Steam account
   1. Go to http://steamcommunity.com/dev/managegameservers and create a GSLT for App ID 730
   2. Save this GSLT in the `private` folder by
      1. Copying `private/.gslt_default` to `private/.gslt`
         1. **Note:** please use these exact names. I've chosen them so that Git
            ignores these files and they always stay local on your server.
      2. Replacing the contents of `private/.gslt` with your GSLT.
4. Run the server inside the docker image: Run `./start_d2.sh`
   1. You can connect to the CSGO server by typing `connect IP_ADDRESS` in CSGO's
      console, where IP_ADDRESS is the IP address of the server.
4. Stop the server from a different terminal on the same computer: Run `./stop.sh` from the `demo_generator` folder.
