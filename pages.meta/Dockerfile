FROM ubuntu
COPY setup_ci_user.sh /root
RUN sh /root/setup_ci_user.sh
USER ci
WORKDIR /home/ci
COPY setup_miniconda.sh /home/ci
RUN sh ~/setup_miniconda.sh
COPY setup_env.sh /home/ci
RUN sh ~/setup_env.sh
