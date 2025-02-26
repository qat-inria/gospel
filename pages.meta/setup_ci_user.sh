set -ex
apt-get update
apt-get upgrade --yes
apt-get install --yes adduser sudo
echo ci ALL=\(ALL\) NOPASSWD:ALL >>/etc/sudoers
adduser --disabled-password --gecos ci --shell /bin/bash ci
