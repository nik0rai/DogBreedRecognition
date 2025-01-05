FROM node:18-slim as base

WORKDIR /app

COPY app/package*.json ./
RUN npm install

COPY app/ ./

EXPOSE 3000

#region Add development with debug

FROM base as dev

CMD [ "npm", "start" ]

#endregion Add development with debug

#region Add production

FROM base as prod

CMD [ "npm", "start" ]

#FROM base as prod

#RUN npm run build

#RUN npm install -g serve

#CMD serve -s build -l 3000

#endregion Add production