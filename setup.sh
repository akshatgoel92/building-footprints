
# Install Python 3.7.0 and make it default version
# First install dependencies 
apt update && apt upgrade -y
apt install build-essential -y
apt install libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev -y
sudo apt-get install zlib1g-dev

# Then get Python .tar file and unzip 
wget https://www.python.org/ftp/python/3.7.0/Python-3.7.0.tgz
tar -xzvf Python-3.7.0.tgz

# Run configuration files
Python-3.7.0/configure --enable-optimizations

# Run make file
make
make install

# Set to system Python
update-alternatives --install /usr/bin/python python /root/python 1

# Go back home
cd ~

# Clone repository
git clone https://github.com/sociometrik/roof-classify.git

# Create a new virtual environment
python -m venv roof-env

# Go into the newly cloned repository
cd ~/roof-classify

# Activate the virtual environment
source ./../roof-env/bin/activate