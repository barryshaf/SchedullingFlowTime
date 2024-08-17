FROM continuumio/miniconda3

WORKDIR /src

COPY . /src

RUN conda install --quiet -y --file conda-requirements.txt
RUN pip install -r pip-requirements.txt

# RUN conda install conda-forge::qiskit-terra
# RUN pip install qiskit-nature==0.4.3

# EXPOSE 8888

# CMD [ "jupyter-lab", "--allow-root", "--no-browser", "--ip=0.0.0.0"]