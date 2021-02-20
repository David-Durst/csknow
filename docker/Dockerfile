FROM cm2network/csgo:latest 
ADD ./update.sh ${STEAMAPPDIR}/csgo.sh
RUN ${STEAMAPPDIR}/csgo.sh
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
#USER root
#RUN apt-get -y update && apt-get -y install less
#USER $USER
