FROM cm2network/csgo:latest 
ENV NONVOLUMESTEAMAPPDIR "${HOMEDIR}/${STEAMAPP}-dedicated-non-volumne"
ADD ./update.sh ${HOMEDIR}/update.sh
RUN ${HOMEDIR}/update.sh
USER root
RUN apt-get -y update && apt-get -y install less procps unzip zip python3
USER $USER
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip
USER root
RUN ./aws/install
USER $USER
ADD ./server.cfg ${NONVOLUMESTEAMAPPDIR}/csgo/cfg/server.cfg
ADD ./gamemode_competitive.cfg ${NONVOLUMESTEAMAPPDIR}/csgo/cfg/gamemode_competitive.cfg
ADD ./gamemode_competitive_server.cfg ${NONVOLUMESTEAMAPPDIR}/csgo/cfg/gamemode_competitive_server.cfg
ADD ./entry.sh ${HOMEDIR}/entry.sh
RUN { \
        echo '@ShutdownOnFailedCommand 1'; \
        echo '@NoPromptForPassword 1'; \
        echo 'login anonymous'; \
        echo 'force_install_dir '"${NONVOLUMESTEAMAPPDIR}"''; \
        echo 'app_update '"${STEAMAPPID}"''; \
        echo 'quit'; \
    } > "${HOMEDIR}/${STEAMAPP}_update.txt"
