# Running VQE-MUB code via Docker containers: Tutorial

Some of the packages required to run the Hamiltonian generation code here can be a tad unstable,
and some require specific OSes to run.

Thus, we built a dockerfile that should create a Jupyter Notebook container with all required packages installed and the relevant source code inserted.

## Pre-requisites: Before Installing Docker
### Windows
Download and install WSL.
Instructions can be found here:
https://learn.microsoft.com/en-us/windows/wsl/install

### MacOS or Linux
No action required.

## Pre-requisites: Install Docker
Instructions can be found here:
https://docs.docker.com/desktop/install


## Steps: Build the docker image
-   In your terminal (bash for Linux, zsh for Mac, PowerShell for Windows), change directory to the root directory of this repo (where you found this file).
-   Run the following command:
```bash

docker build . -t dqes
```

## Steps: Run the docker image
-   By running the image we built, we are creating a container: a lightweight Virtual Machine running the Jupyter Notebook server.
-   Run the following command:

```bash

docker run -it -p 8888:8888 dqes
```

In the terminal, you will see something similar to this:
```
    To access the server, open this file in a browser:
        file:///root/.local/share/jupyter/runtime/jpserver-1-open.html
    Or copy and paste one of these URLs:
        http://f6dd1e6862d3:8888/lab?token=49324ce7baf37e2a60ddda1df9778d50fa2f441ef3f0004a
        http://127.0.0.1:8888/lab?token=49324ce7baf37e2a60ddda1df9778d50fa2f441ef3f0004a

```

Copy and paste the link at the bottom to your browser to start working.
If you are not familiar with python notebooks or JupyterLab, I recommend watching a tutorial or two - I promise it's not difficult.
That's about it!