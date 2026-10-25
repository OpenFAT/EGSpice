#!/usr/bin/env bash
# Install each distro package in a clean container of its distro and run
# release_smoke.sh (a built-in model and runtime Verilog-A) on /usr/bin/espice.
# On amd64 the AUR PKGBUILD is also built by makepkg from the local .deb.
# GPU packages run the same checks on the CPU backend (no GPU on the runner):
# their espice must start without a driver, which it dlopens only when asked.
# Usage: tools/distro_smoke.sh DIST VERSION amd64|arm64 FIXTURES [espice|espice-cuda|espice-hip]
set -euo pipefail
dist=$(cd "$1" && pwd) version=$2 arch=$3 fixtures=$(cd "$4" && pwd) name=${5:-espice}
root=$(cd "$(dirname "$0")/.." && pwd)
stem=/dist/$name-$version-linux-$arch
smoke='bash /smoke.sh /usr/bin/espice /fixtures'
# A GPU package must carry its backend: `espice --version` lists the built-in ones.
case $name in
  espice-cuda) smoke="espice --version | grep 'backends:.*cuda' >/dev/null; $smoke" ;;
  espice-hip) smoke="for b in /usr/lib/espice/bin/espice-gfx*; do \$b --version | grep 'backends:.*hip' >/dev/null; done; $smoke" ;;
esac
aur=$(mktemp -d)
trap 'rm -rf "$aur"' EXIT
python3 "$root/tools/release.py" pkgbuild "$dist" "$version" > "$aur/PKGBUILD"

run() {
  local image=$1 script=$2
  echo "::group::$image"
  docker run --rm -v "$dist:/dist:ro" -v "$fixtures:/fixtures:ro" -v "$aur:/aur:ro" \
    -v "$root/tools/release_smoke.sh:/smoke.sh:ro" "$image" bash -euo pipefail -c "$script"
  echo "::endgroup::"
}

run docker.io/library/ubuntu:24.04 "dpkg -i $stem.deb; $smoke"
run docker.io/library/debian:12 "dpkg -i $stem.deb; $smoke"
run docker.io/library/fedora:42 "rpm -i $stem.rpm; $smoke"
run docker.io/opensuse/leap:15.6 "rpm -i $stem.rpm; $smoke" # ships awk: no network needed
# Arch Linux publishes x86_64 images only; the AUR PKGBUILD is espice's.
[ "$arch" = amd64 ] && run docker.io/library/archlinux:latest "pacman -U --noconfirm $stem.pkg.tar.zst; $smoke"
if [ "$arch" = amd64 ] && [ "$name" = espice ]; then
  # makepkg uses a source file already beside the PKGBUILD instead of downloading it.
  run docker.io/library/archlinux:latest "
    pacman -Sy --noconfirm --needed fakeroot >/dev/null
    useradd -m build
    cp /aur/PKGBUILD $stem.deb /home/build/
    chown -R build /home/build
    su build -c 'cd /home/build && makepkg --nodeps --noconfirm'
    pacman -U --noconfirm /home/build/espice-bin-$version-1-x86_64.pkg.tar.zst
    $smoke"
fi
