# Setting the accompanion in a fresh VM machine using Multipass

- Create a machine
- install dependencies
- install the Conda environment
- run accompanion

## Create a VM

After you installed multipass at your system you need to create a new VM with internet access.
To find your Wifi networks use `mutlipass networks list`. Usually the wireless network name is WiFi.


```shell
multipass launch -c 4 -d 12G -m 6G -n accompanion --network WiFi
```

When your new VM is installed run a shell on it using `mutlipass shell accompanion` and then you can update and upgrade your system:
```shell
sudo apt update && sudo apt upgrade -y
```

## Install dependences

You will need to install some dependencies for the accompanion packages:
```shell
sudo apt install git
sudo apt-get install gcc
sudo apt-get install libjack-dev
sudo apt-get install libasound2-dev
sudo apt install g++
```
This dependencies address C++ dependable libraries but mainly _RTMIDI_.



Then let's install conda:
```shell
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
export PATH="/home/ubuntu/miniconda/bin:$PATH"
rm ./Miniconda3-latest-Linux-x86_64.sh
bash
```

#### Installing a GUI for your VM (Optional)

On a shell of your accompanion VM run:
```shell
sudo apt install -y tasksel
sudo tasksel
```
Install XUbuntu Minimal installation (XFC).

The VM doesn't have a GUI for itself so you will need to ssh to it via your local windows machine. To allow remote desktop access to the virtual machine you will need to install the remote desktop server on the VM.
```shell
sudo apt install -y xrdp
```
To find your ip just type `ip addr`.
You can access it using windows remote desktop tool.


## Install requirements

You can clone the repository, recommended to do so in the `/home/ubuntu` directory (_ubuntu_ is the default sudo user). 

If you want to have access to the accompanion directory from both your local machine and the VM you can mount the accompanion.
You can clone the accompanion with your windows machine and then mount the folder by running:
```shell
multipass mount "C:/Users/Username/Directory/to/accompanion" "/home/ubuntu/"
```
Now your folder is accesible from both your local windows machine and your VM.


To install the requirements
```shell
cd /home/ubuntu/accompanion
conda env create -f environment.yml
conda activate accompanion
pip install -e .
```

## Run Accompanion

To run the ACCompanion:
```shell
cd /home/ubuntu/accompanion
python ./bin/launch_acc.py
```