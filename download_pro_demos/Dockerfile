FROM node:15.14.0-buster

RUN echo "deb http://http.us.debian.org/debian stable main contrib non-free" >> /etc/apt/sources.list
RUN apt-get -y update && apt-get -y install less procps unzip zip python3 p7zip p7zip-rar

WORKDIR /go/src/app
COPY . .

RUN npm install

CMD ["npm", "run", "scrape"]
