# Use an official Python runtime as a parent image -----------------------------
FROM python:3.10 AS installer
# Create virtual environment for the app ---------------------------------------
RUN python3 -m venv "venv"
ENV PATH='/venv/bin:$PATH'
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=../requirements.txt,target=requirements.txt \
    python -m pip install --no-cache -r requirements.txt
# ==============================================================================
# Run image --------------------------------------------------------------------
FROM python:3.10-slim
# Install system dependencies --------------------------------------------------
# RUN apt update && apt install -y coreutils && apt clean && rm -rf /var/lib/apt/lists/*
RUN apt update && apt upgrade --yes
RUN apt install --yes coreutils
RUN apt clean && apt autoclean && apt autoremove
# Copy python virtual environment ----------------------------------------------
COPY --from=installer "/venv" "/venv"
ENV PATH='/venv/bin:$PATH'
# Incorporate custom code ------------------------------------------------------
ENV APPDIR=/usr/src/app
WORKDIR ${APPDIR}
COPY ../ .
# Add non-privileged user ------------------------------------------------------
ENV APP_USER=app_runner
ENV APP_UID=1000
RUN echo "${APP_USER}:x:${APP_UID}:${APP_UID}:app user:/nonexistent:/sbin/nologin" >> /etc/passwd && \
    echo "${APP_USER}:x:${APP_UID}:" >> /etc/group
# Prepare directory structure --------------------------------------------------
ENV LOGS_DIR=${APPDIR}/logs
ENV DB_DIR=${APPDIR}/db
RUN python -c "import os; os.makedirs('${LOGS_DIR}', exist_ok=True); os.makedirs('${DB_DIR}', exist_ok=True)"
RUN python -c "import os; import pwd, grp; \
uid = pwd.getpwnam('app_runner').pw_uid; \
gid = grp.getgrnam('app_runner').gr_gid; \
[os.chown(os.path.join(root, f), uid, gid) for d in ('/usr/src/app/logs', '/usr/src/app/db') \
for root, dirs, files in os.walk(d) for f in dirs + files] + \
[os.chown(d, uid, gid) for d in ('/usr/src/app/logs', '/usr/src/app/db')]"
# Run the application ----------------------------------------------------------
ENV PYTHONBUFFERED=1
ENV PYTHONDONTWRTITEBYTECODE=1
USER ${APP_USER}
ENTRYPOINT ["python", "-m", "src"]