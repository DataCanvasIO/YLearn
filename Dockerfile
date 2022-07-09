From python:3.8.13-buster

#ARG PIP_PKGS="numpy pandas scikit-learn torch lightgbm igraph jupyterlab ipywidgets tqdm ipywidgets matplotlib shap"
ARG PIP_PKGS="ylearn lightgbm igraph jupyterlab ipywidgets tqdm ipywidgets matplotlib shap"
ARG PIP_OPTS="--disable-pip-version-check --no-cache-dir"
#ARG PIP_OPTS="--disable-pip-version-check --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple/"

# COPY sources.list /etc/apt/sources.list

RUN apt update \
    && apt install -y graphviz \
    && apt clean \
    && pip install $PIP_OPTS $PIP_PKGS \
    && v=$(pip show ylearn|awk '/Version/{print($2)}') \
    && echo ylearn version:$v \
    && pip download --no-deps --dest /tmp/ $PIP_OPTS ylearn==$v \
    && tar xzf /tmp/ylearn-$v.tar.gz -C /tmp/ \
    && mkdir -p /opt/datacanvas \
    && cp -r /tmp/ylearn-$v/example_usages /opt/datacanvas/ \
    && echo "#!/bin/bash\njupyter lab --notebook-dir=/opt/datacanvas --ip=0.0.0.0 --port=\$NotebookPort --no-browser --allow-root --NotebookApp.token=\$NotebookToken" > /entrypoint.sh \
    && chmod +x /entrypoint.sh \
    && rm -rf /var/lib/apt/lists \
    && rm -rf /var/cache/* \
    && rm -rf /var/log/* \
    && rm -rf /root/.cache \
    && rm -rf /tmp/*

EXPOSE 8888

ENV NotebookToken="" \
    NotebookPort=8888

CMD ["/entrypoint.sh"]

# docker run --rm --name ylearn -p 8888:8888 -e NotebookToken=your-token  datacanvas/ylearn
