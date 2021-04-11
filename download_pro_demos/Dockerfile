FROM node:15.14.0-buster

RUN echo "deb http://http.us.debian.org/debian stable main contrib non-free" >> /etc/apt/sources.list
RUN apt-get -y update && apt-get -y install less procps unzip zip python3 p7zip p7zip-rar

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip
RUN ./aws/install

WORKDIR /go/src/app
COPY . .

RUN npm install

CMD ["npm", "run", "scrape"]
