FROM python:3.12-slim AS player
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
RUN pip install --no-cache-dir websockets==16.0
COPY integrations/__init__.py /app/integrations/__init__.py
COPY integrations/softmax/__init__.py integrations/softmax/protocol.py integrations/softmax/player.py integrations/softmax/builder_player.py /app/integrations/softmax/
COPY competition/agents/expander_python /app/competition/agents/expander_python
# The runner mounts /tmp destinations. Code and dependencies need no write access.
USER 10001:10001
CMD ["python", "-m", "integrations.softmax.player"]

FROM python:3.12-slim AS game
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
WORKDIR /app
COPY integrations/softmax/requirements.lock /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY generals /app/generals
COPY integrations/__init__.py /app/integrations/__init__.py
COPY integrations/softmax/__init__.py integrations/softmax/config.py integrations/softmax/protocol.py integrations/softmax/engine.py integrations/softmax/artifacts.py integrations/softmax/server.py /app/integrations/softmax/
COPY integrations/softmax/static /app/integrations/softmax/static
EXPOSE 8080
CMD ["python", "-m", "integrations.softmax.server"]

FROM game AS hunter
COPY integrations/softmax/hunter_player.py /app/integrations/softmax/
USER 10001:10001
CMD ["python", "-m", "integrations.softmax.hunter_player"]
