FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
RUN apt-get -y update && \
    apt-get -y install less neovim mesa-utils
ARG PUID=1000
ENV USER steam
ENV HOMEDIR "/home/${USER}"
RUN useradd -u "${PUID}" -m ${USER}
USER steam
WORKDIR /home/steam
CMD ["bash", "entry.sh"]
