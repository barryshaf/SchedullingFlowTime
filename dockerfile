FROM continuumio/miniconda3

WORKDIR /src

RUN conda install -y conda-forge::pyscf
RUN conda install -y jupyter
RUN conda install -y matplotlib

RUN pip install qiskit-nature==0.7.2


EXPOSE 8888

CMD [ "jupyter-lab", "--allow-root", "--no-browser", "--ip=0.0.0.0"]