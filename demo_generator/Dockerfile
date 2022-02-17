FROM cm2network/csgo:latest 
ENV NONVOLUMESTEAMAPPDIR "${HOMEDIR}/${STEAMAPP}-dedicated-non-volumne"
ADD ./update.sh ${HOMEDIR}/update.sh
RUN ${HOMEDIR}/update.sh
USER root
# DONT INSTALL c++, g++ works but c++ crashes for some reason
RUN apt-get -y update && \
    apt-get -y install less procps unzip zip python3 jq libgomp1 make g++ git
RUN wget https://github.com/Kitware/CMake/releases/download/v3.22.2/cmake-3.22.2-linux-x86_64.sh && \
    chmod a+x cmake-3.22.2-linux-x86_64.sh && \
    mkdir cmake && \
    ./cmake-3.22.2-linux-x86_64.sh --skip-license --prefix=cmake && \
    update-alternatives --install /usr/bin/cmake cmake ${HOMEDIR}/cmake/bin/cmake 60
    #&& \
#    ln cmake/bin/cmake /usr/bin/cmake
USER $USER
RUN git clone https://github.com/David-Durst/csknow.git 
# && \
    #cd csknow/analytics 
    #&& \
    #./scripts/bot_build.sh
#RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
#    unzip awscliv2.zip
#USER root
#RUN ./aws/install
#USER $USER
#ADD ./install_sourcemod.sh ${HOMEDIR}/install_sourcemod.sh
#RUN ${HOMEDIR}/install_sourcemod.sh ${NONVOLUMESTEAMAPPDIR}/csgo
#ADD ./internal_scripts/server.cfg ${NONVOLUMESTEAMAPPDIR}/csgo/cfg/server.cfg
#ADD ./internal_scripts/gamemode_competitive.cfg ${NONVOLUMESTEAMAPPDIR}/csgo/cfg/gamemode_competitive.cfg
#ADD ./internal_scripts/gamemode_competitive_server.cfg ${NONVOLUMESTEAMAPPDIR}/csgo/cfg/gamemode_competitive_server.cfg
#ADD ./internal_scripts/gamemode_casual_server.cfg ${NONVOLUMESTEAMAPPDIR}/csgo/cfg/gamemode_casual_server.cfg
#ADD ./internal_scripts/run_csgo_bots.sh ${NONVOLUMESTEAMAPPDIR}/docker_run_csgo_bots.sh
#ADD ./internal_scripts/run_csgo_bots_2v1.sh ${NONVOLUMESTEAMAPPDIR}/docker_run_csgo_bots_2v1.sh
#ADD ./entry.sh ${HOMEDIR}/entry.sh
#ADD ./tmp/bot_run_no_build.sh ${NONVOLUMESTEAMAPPDIR}/bot_run_no_build.sh
#ADD ./tmp/csknow_bot ${NONVOLUMESTEAMAPPDIR}/csknow_bot
#ADD ./maps/bot_playground.bsp ./maps/bot_playground.nav \
#    ./maps/bot_playground_2v1.bsp ./maps/bot_playground_2v1.nav ${NONVOLUMESTEAMAPPDIR}/csgo/maps/
#ADD ./upload_logs.py ${HOMEDIR}/upload_logs.py
#RUN { \
#        echo '@ShutdownOnFailedCommand 1'; \
#        echo '@NoPromptForPassword 1'; \
#        echo 'force_install_dir '"${NONVOLUMESTEAMAPPDIR}"''; \
#        echo 'login anonymous'; \
#        echo 'app_update '"${STEAMAPPID}"''; \
#        echo 'quit'; \
#    } > "${HOMEDIR}/${STEAMAPP}_update.txt"
#CMD ["bash", "entry.sh"]
