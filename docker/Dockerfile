FROM cm2network/csgo:latest 
ENV NONVOLUMESTEAMAPPDIR "${HOMEDIR}/${STEAMAPP}-dedicated-non-volumne"
ADD ./update.sh ${HOMEDIR}/update.sh
RUN ${HOMEDIR}/update.sh
USER root
RUN apt-get -y update && apt-get -y install less procps
USER $USER
ADD ./server.cfg ${NONVOLUMESTEAMAPPDIR}/csgo/cfg/server.cfg
ADD ./gamemode_competitive.cfg ${NONVOLUMESTEAMAPPDIR}/csgo/cfg/gamemode_competitive.cfg
ADD ./entry.sh ${HOMEDIR}/entry.sh
RUN { \
        echo '@ShutdownOnFailedCommand 1'; \
        echo '@NoPromptForPassword 1'; \
        echo 'login anonymous'; \
        echo 'force_install_dir '"${NONVOLUMESTEAMAPPDIR}"''; \
        echo 'app_update '"${STEAMAPPID}"''; \
        echo 'quit'; \
    } > "${HOMEDIR}/${STEAMAPP}_update.txt" \ 


#RUN /home/csgo/hlserver/update.sh
#ADD ./csgo.sh /home/csgo/hlserver/csgo.sh
#RUN echo "tv_enable 1" >> /home/csgo/hlserver/csgo/csgo/cfg/autoexec.cfg
#RUN echo "tv_autorecord 1" >> /home/csgo/hlserver/csgo/csgo/cfg/autoexec.cfg
#RUN echo "bot_quota 10" >> /home/csgo/hlserver/csgo/csgo/cfg/server.cfg
#RUN echo "bot_quota_mode fill" >> /home/csgo/hlserver/csgo/csgo/cfg/server.cfg
#RUN echo "bot_difficulty 3" >> /home/csgo/hlserver/csgo/csgo/cfg/server.cfg
#RUN echo "bot_join_after_player 0" >> /home/csgo/hlserver/csgo/csgo/cfg/server.cfg
#RUN echo "mp_maxrounds 2" >> /home/csgo/hlserver/csgo/csgo/cfg/server.cfg
#RUN echo "mp_roundtime 0.5" >> /home/csgo/hlserver/csgo/csgo/cfg/server.cfg
#ADD ./gamemode_competitive.cfg /home/csgo/hlserver/csgo/csgo/cfg/gamemode_competitive.cfg
