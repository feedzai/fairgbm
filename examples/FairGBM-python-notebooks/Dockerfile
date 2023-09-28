FROM --platform=linux/amd64 ubuntu:latest

LABEL description="Starts a jupyter notebook server with fairgbm set-up."

ENV JUPYTER_PORT 8888

COPY * /fairgbm-notebooks/

RUN apt-get update && apt-get install -y wget
RUN apt-get update && apt-get install vim -y

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p $HOME/miniconda

ENV PATH="/root/miniconda/bin:${PATH}"

RUN conda install -y numpy pandas jupyter

WORKDIR /fairgbm-notebooks

EXPOSE 8888/tcp

RUN python -m pip install fairgbm
RUN python -m pip install -r requirements.txt

RUN python -m jupyter trust *.ipynb

CMD python -m jupyter notebook --port=${JUPYTER_PORT} --ip '0.0.0.0' --allow-root --no-browser
