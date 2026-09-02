# Build stage: toolchain + git only live here; the final image just gets the
# populated venv.
FROM python:3.11-slim AS build
RUN apt-get update && apt-get install --no-install-recommends --yes \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Clone and install leds and its dependencies into the venv. A full clone
# gives setuptools_scm the git history it needs to derive the version. Override
# LEDS_REF to build a specific tag/branch/commit.
ARG LEDS_REPO=https://github.com/legend-exp/leds.git
ARG LEDS_REF=main
RUN git clone "${LEDS_REPO}" /src \
    && git -C /src checkout "${LEDS_REF}" \
    && uv pip install --no-cache /src

FROM python:3.11-slim
COPY --from=build /opt/venv /opt/venv

# The array view renders through matplotlib (Agg, head-less) and Bokeh writes
# caches at runtime. Point caches/HOME at writable /tmp so the container works
# under an arbitrary non-root UID, which Spin assigns via runAsUser.
ENV PATH="/opt/venv/bin:$PATH" \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/mpl \
    HOME=/tmp \
    PYTHONUNBUFFERED=1 \
    PORT=5006 \
    NUM_PROCS=2

# Runtime configuration is supplied by the Spin service, not baked in:
#   LEDS_BASE_PATH        production-cycle directory (a mounted NERSC global
#                         filesystem, e.g. CFS); `leds serve` falls back to it.
#   BOKEH_ALLOW_WS_ORIGIN comma-separated public host[:port] allowed to open a
#                         websocket (the Spin hostname). Bokeh reads this env
#                         var directly; required when proxied behind Spin's LB.
#   NUM_PROCS             Panel server processes. Sessions in one process share
#                         its event loop, so one user's blocking file read
#                         stalls the others; more processes isolate them. They
#                         do NOT share caches, so the memory bounds below are
#                         per process -- multiply by this when sizing limits.
#   LEDS_PREWARM          build the newest channelmap before forking, so each
#                         worker's first session is warm (delays listening).
#   LEDS_CACHE_TTL        seconds before metadata-derived caches are re-read
#                         (default 3600); bounds how stale an updated metadata
#                         checkout can be without a restart.
#   LEDS_SCAN_TTL         seconds before directory scans are redone (300).
#   LEDS_MAX_CACHED_RUN_SPECTRA
#                         whole runs of per-hit energies held per worker (4).
#                         The dominant memory line -- lower it if the container
#                         is tight, raise it if users hop between runs.
#
# Login (all optional; secrets should come from Spin secrets, not the image):
#   LEDS_COOKIE_SECRET    signs the auth cookie; set a fixed value so logins
#                         survive restarts and work across NUM_PROCS workers.
#   LEDS_BASIC_AUTH       shared password or path to a {user: password} JSON
#                         file; ignored when LDAP is configured.
#   LEDS_LDAP_SERVER      enables LDAP login, e.g. ldaps://ldap.example:636.
#     Direct bind:        LEDS_LDAP_USER_DN_TEMPLATE
#                         (e.g. "uid={username},ou=people,dc=legend,dc=org";
#                         takes precedence over search-then-bind).
#     Search-then-bind:   LEDS_LDAP_BIND_DN + LEDS_LDAP_BIND_PASSWORD +
#                         LEDS_LDAP_SEARCH_BASE, with optional
#                         LEDS_LDAP_USER_FILTER (default "(uid={username})").
#     LEDS_LDAP_GROUP_DN  if set, only members of this group (member/
#                         uniqueMember/memberUid) may log in.
#     LEDS_LDAP_STARTTLS  "1" upgrades an ldap:// connection via StartTLS.
#     LEDS_LDAP_CA_FILE   CA bundle path (e.g. a mounted secret); defaults to
#                         certifi's bundle.
EXPOSE 5006
CMD ["sh", "-c", "leds serve --address 0.0.0.0 --port ${PORT} --num-procs ${NUM_PROCS}"]
