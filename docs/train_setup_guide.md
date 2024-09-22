# Developer Setup Guide
These are instructions for jetting up development on CSGO's data analytics and bot systems.
If you just want to play against the bots, see my
[user instructions](https://github.com/David-Durst/csknow/blob/master/docs/bot_user_instructions.md).

## Prerequisites
These instructions assume Ubuntu 22.04 OS with sufficient space to install a
CSGO server (~30 GB). If your personal computer doesn't match these
specifications, I recommend creating a machine on AWS using [my
guide](https://github.com/David-Durst/csknow/blob/master/docs/aws_machine_creation_guide.md).

## Server Setup
1. Make sure you are using a user named steam `sudo usermod -a -G sudo steam` and do all further instructions on their account.
1. Checkout this repository in the home folder of the CSGO server.
2. Copy the update and run scripts to the home folder: `cd ~; cp csknow/demo_generator/no_docker_scripts/*.sh .` 
3. Install csgo by running `./update.sh`
4. Create a GLST token by following the instructions [here](https://steamcommunity.com/dev/managegameservers)
5. Copy the server config `cp csknow/demo_generator/no_docker_scripts/server.cfg csgo-ds/csgo/cfg/`
6. Update the server config with your GSLT, server password, and rcon password `vim csgo-ds/csgo/cfg/server.cfg`
7. Install Sourcemod and Metamod
    1. Go to the parent folder of the CSGO addons `cd ~/csgo-ds/csgo`
    2. Download and install Metamod - `https://mms.alliedmods.net/mmsdrop/1.11/mmsource-1.11.0-git1153-linux.tar.gz; tar -xzf mmsource-1.11.0-git1153-linux.tar.gz`
    2. Download and install Sourcemod - `https://sm.alliedmods.net/smdrop/1.11/sourcemod-1.11.0-git6949-linux.tar.gz; tar -xzf sourcemod-1.11.0-git6949-linux.tar.gz`
3. Download and install the lobby patch
    1. Checkout the code `cd ~; git clone https://github.com/eldoradoel/NoLobbyReservation`
    2. Install the gamedata part of the patch `cp NoLobbyReservation/csgo/addons/sourcemod/gamedata/nolobbyreservation.games.txt csgo-ds/csgo/addons/sourcemod/gamedata`
    3. Install the scripting part of the patch `cp NoLobbyReservation/csgo/addons/sourcemod/scripting/nolobbyreservation.sp csgo-ds/csgo/addons/sourcemod/scripting; ./compile.sh nolobbyreservation.sp ; cp compiled/nolobbyreservation.smx ../plugins`
4. Download and install the bot-link code
    1. Checkout the code `cd ~; git clone https://github.com/David-Durst/bot-link.git`
    2. Symlink the code in place `ln -s bot-link ~/csgo-ds/csgo/addons/sourcemod/scripting/bot-link`
    3. Deploy the link code `cd ~/csgo-ds/csgo/addons/sourcemod/scripting/bot-link; ./bot-link/deploy.sh link`
5. Run the server `cd ~; ./run_csgo.sh`

