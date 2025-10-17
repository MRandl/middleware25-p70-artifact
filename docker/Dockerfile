FROM ubuntu:22.04

ARG LDAP_USERNAME
ARG LDAP_UID
ARG LDAP_GROUPNAME
ARG LDAP_GID
RUN groupadd ${LDAP_GROUPNAME} --gid ${LDAP_GID}
RUN useradd -p '$6$4tHtCPAtD7LTM7Yq$QXr12iwYTWKCY2kYsC3f1AhTXrPGZWOOVARSPorpq4tnOH/Q2dCnIY9S04y3XUgX2ZxzNSZsbWdIUINtsRSe5.' -m -s /bin/bash -g ${LDAP_GROUPNAME} -u ${LDAP_UID} ${LDAP_USERNAME}

RUN apt update && apt install -y python3.10 python3-pip python3-venv \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

RUN apt update && apt install -y curl libsodium-dev && pip install transformers==4.43.1 datasets==2.20.0 evaluate==0.4.0 wget==3.2 rouge_score==0.1.2 faiss-cpu==1.7.4 sentence-transformers==2.2.2 pyserini==0.19.0 flask==2.2.5 gunicorn openai==0.27.8 nltk accelerate jsonlines protobuf==3.20.* "numpy<2" "huggingface_hub==0.25.2" maturin
RUN apt update && apt install -y make cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev sudo tmux nano wget curl git git-lfs unzip
RUN adduser ${LDAP_USERNAME} sudo


# Set the working directory in your user's home
WORKDIR /home/${LDAP_USERNAME}
COPY ../* medrag
USER ${LDAP_USERNAME}
RUN curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain nightly -y
ENV PATH="/home/${LDAP_USERNAME}/.cargo/bin:${PATH}"

ENTRYPOINT ["/bin/bash", "-c", "sleep infinity"]
