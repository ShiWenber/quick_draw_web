FROM openjdk:11-jre-slim
VOLUME /tmp
ARG JAVA_OPTS
ENV JAVA_OPTS=$JAVA_OPTS
# change to UTC+8 java's jvm has its own timezone config, so the system timezone will not affect the java program
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
COPY *.jar app.jar
EXPOSE 9527
# ENTRYPOINT exec java $JAVA_OPTS -jar dockerbootoj.jar
# For Spring-Boot project, use the entrypoint below to reduce Tomcat startup time. you should use the '-D user.timezone=GMT+08' to config the jvm's timezone
ENTRYPOINT exec java $JAVA_OPTS -Djava.security.egd=file:/dev/./urandom -D user.timezone=GMT+08 -jar app.jar
