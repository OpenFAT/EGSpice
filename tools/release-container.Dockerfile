# nix-bundle's archive launcher uses the host shell, tar and bzip2 before
# entering its bundled Nix closure. The actual smoke run has no network.
FROM docker.io/library/ubuntu:24.04
RUN apt-get update \
    && apt-get install -y --no-install-recommends bzip2 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home bundle
USER bundle
WORKDIR /home/bundle
