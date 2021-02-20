FROM gonzih/csgo-server:latest
ADD ./csgo.sh /home/csgo/hlserver/csgo.sh
RUN /home/csgo/hlserver/update.sh
RUN echo "tv_enable 1" >> /home/csgo/hlserver/csgo/csgo/cfg/autoexec.cfg
RUN echo "tv_autorecord 1" >> /home/csgo/hlserver/csgo/csgo/cfg/autoexec.cfg
RUN echo "bot_quota 10" >> /home/csgo/hlserver/csgo/csgo/cfg/server.cfg
RUN echo "bot_quota_mode fill" >> /home/csgo/hlserver/csgo/csgo/cfg/server.cfg
RUN echo "bot_difficulty 3" >> /home/csgo/hlserver/csgo/csgo/cfg/server.cfg
RUN echo "mp_maxrounds 2" >> /home/csgo/hlserver/csgo/csgo/cfg/server.cfg
RUN echo "mp_roundtime 0.5" >> /home/csgo/hlserver/csgo/csgo/cfg/server.cfg

