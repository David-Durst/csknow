FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
RUN dpkg --add-architecture i386 
RUN apt-get -y update && \
    DEBIAN_FRONTEND=noninteractive apt-get -yq install less neovim mesa-utils libc6-i386 libgl1-mesa-dri:i386 libgl1:i386 libgl1-mesa-glx:i386 gnome 
ARG PUID=1000
ENV USER steam
ENV HOMEDIR "/home/${USER}"
RUN useradd -u "${PUID}" -m ${USER}
USER steam
WORKDIR /home/steam
USER root
ADD game/steam_latest.deb ${HOMEDIR}/steam_latest.deb
RUN apt install ${HOMEDIR}/steam_latest.deb -y
run apt-get install -y --no-install-recommends \
    ca-certificates \
# zenity is used during early startup for dialogs and progress bars
		zenity \
# wget is used for uploading crash dumps
		wget \
# needed for "dbus-launch" (not sure what that's used for)
		dbus-x11 \
# needed for "localedef" below
		locales
RUN localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG en_US.utf8
RUN apt-get -y install libfontconfig1:i386
RUN apt-get install -y kmod mesa-utils
RUN adduser steam video
RUN apt-get -y update && apt-get install -y libc6:amd64 libc6:i386 libegl1:amd64 libegl1:i386 libgbm1:amd64 libgbm1:i386 libgl1-mesa-dri:amd64 libgl1-mesa-dri:i386 libgl1:amd64 libgl1:i386 steam-libs-amd64:amd64 steam-libs-i386:i386
#RUN add-apt-repository multiverse
#RUN apt install -y steam
#ADD steam.desktop /usr/share/applications/steam.desktop


USER steam
CMD ["bash", "entry.sh"]
