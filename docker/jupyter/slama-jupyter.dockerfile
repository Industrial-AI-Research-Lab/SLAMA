FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
RUN pip install jupyterhub==5.2.1

ARG NB_USER="jovyan"
ARG NB_UID="1000"
ARG NB_GID="100"

USER root

# Install all OS dependencies for the Server that starts
# but lacks all features (e.g., download as all possible file formats)
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    # - Add necessary fonts for matplotlib/seaborn
    #   See https://github.com/jupyter/docker-stacks/pull/380 for details
    fonts-liberation \
    # - `pandoc` is used to convert notebooks to html files
    #   it's not present in the aarch64 Ubuntu image, so we install it here
    pandoc \
    # - `run-one` - a wrapper script that runs no more
    #   than one unique instance of some command with a unique set of arguments,
    #   we use `run-one-constantly` to support the `RESTARTABLE` option
    run-one && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install jupyterlab

USER ${NB_UID}

# Install JupyterHub, JupyterLab, NBClassic and Jupyter Notebook
# Generate a Jupyter Server config
# Cleanup temporary files
# Correct permissions
# Do all this in a single RUN command to avoid duplicating all of the
# files across image layers when the permissions change
WORKDIR /tmp

ENV JUPYTER_PORT=8888
EXPOSE $JUPYTER_PORT

# Configure container startup
# CMD ["start-notebook.py"]

# Copy local files as late as possible to avoid cache busting
COPY start-notebook.py start-notebook.sh start-singleuser.py start-singleuser.sh /usr/local/bin/
COPY jupyter_server_config.py docker_healthcheck.py /etc/jupyter/

# Fix permissions on /etc/jupyter as root
USER root

RUN chmod 777 /usr/local/bin/start-notebook.py
RUN chmod 777 /usr/local/bin/start-notebook.sh
RUN chmod 777 /usr/local/bin/start-singleuser.py
RUN chmod 777 /usr/local/bin/start-singleuser.sh

# Copy a script that we will use to correct permissions after running certain commands
COPY fix-permissions /usr/local/bin/fix-permissions
RUN chmod a+rx /usr/local/bin/fix-permissions
RUN fix-permissions /etc/jupyter/
RUN fix-permissions /usr/local/bin/

# HEALTHCHECK documentation: https://docs.docker.com/engine/reference/builder/#healthcheck
# This healtcheck works well for `lab`, `notebook`, `nbclassic`, `server`, and `retro` jupyter commands
# https://github.com/jupyter/docker-stacks/issues/915#issuecomment-1068528799
HEALTHCHECK --interval=3s --timeout=1s --start-period=3s --retries=3 \
    CMD /etc/jupyter/docker_healthcheck.py || exit 1

# Switch back to jovyan to avoid accidental container runs as root
USER ${NB_UID}

ENV HOME="/home/${NB_USER}"

WORKDIR "${HOME}"


CMD ["jupyter", "lab", "--notebook-dir=/home/jovyan", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''", "--NotebookApp.allow_origin='*'"]
