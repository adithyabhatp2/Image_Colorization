mkdir ~/server-setup
cd ~/server-setup

sudo apt-get update
sudo apt-get --assume-yes upgrade
sudo apt-get --assume-yes install tmux build-essential gcc g++ make binutils libcupti-dev
sudo apt-get --assume-yes install software-properties-common
sudo apt-get --assume-yes install unzip

# CUDA
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
sudo apt-get update
sudo apt-get -y install cuda
sudo modprobe nvidia
nvidia-smi

# Anaconda - python2
wget https://repo.continuum.io/archive/Anaconda2-4.3.1-Linux-x86_64.sh
bash Anaconda2-4.3.1-Linux-x86_64.sh -b
echo "export PATH=\"$HOME/anaconda2/bin:\$PATH\"" >> ~/.bashrc
export PATH="$HOME/anaconda2/bin:$PATH"
conda update python -y
conda upgrade -y --all

# Keras
pip install keras

# CUDNN
wget "http://platform.ai/files/cudnn.tgz" -O "cudnn.tgz"
tar -zxf cudnn.tgz
cd cuda
sudo cp lib64/* /usr/local/cuda/lib64/
sudo cp include/* /usr/local/cuda/include/
echo "export LD_LIBRARY_PATH=/usr/local/cuda/include/:$LD_LIBRARY_PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH" >> ~/.bashrc

cd ..

# Jupyter Notebook setup
jupyter notebook --generate-config
echo "
c.NotebookApp.password = u'sha1:63f7f9b3a7bc:0aa31e5953482e63193a25d41abae90fd6b72993'
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
" >> $HOME/.jupyter/jupyter_notebook_config.py

sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 8888

# Tensorflow GPU - py2.7
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.1-cp27-none-linux_x86_64.whl


# cs838 Deep Learning Lab3 stuff
mkdir ~/lab3
cd ~/lab3
wget http://pages.cs.wisc.edu/~shavlik/cs638/Lab3/images.zip
unzip images.zip
wget https://gist.githubusercontent.com/dsesclei/d7b60e84aa4d4a3374cbf8ab671ab6d8/raw/639471c1334b76cc4b39d2aebf4d247a29441d7e/Lab3.ipynb