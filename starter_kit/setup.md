# Setup SMARTS Environment

## Setup from Dockerfile

We recommend use docker to setup SMARTS envirionment. A Dockerfile which contains dependencies of SMARTS is supplied in 
your starter-kit. Follow the example command below and create your instance.

```bash
# unzip the starter_kit and place somewhere convenient on your machine. (e.x. ~/dai/track1/starter_kit)

# get docker
docker build -t smarts/dai .
# create docker container
docker run -itd -p 6006:6006 -p 8081:8081 -p 8082:8082 -v ~/dai:~/dai --name smarts smarts/dai bash
# get into the docker
docker exec -it smarts bash

# go to working dir
cd ~/dai/track1/starter_kit
# install the dependencies in the docker
pip install smarts-0.3.7-py3-none-any.whl
pip install smarts-0.3.7-py3-none-any.whl[train]
pip install smarts-0.3.7-py3-none-any.whl[dev]

# download the public datasets from Codalab to ../dataset_public

# test that the sim works
python train_example/keeplane_example.py --scenario ../dataset_public/simple_loop/simpleloop_a
```

## Setup from scratch

To setup the environment for SMARTS, run the following commands. Currently SMARTS can be setup in MAC OS system and Ubuntu System. If you are using Windows system, try docker for windows.


```bash

# unzip the starter_kit and place somewhere convenient on your machine. (e.x. ~/dai/track1/starter_kit)

cd ~/dai/track1/starter_kit
./install_deps.sh
# ...and, follow any on-screen instructions

# test that the sumo installation worked, skip this if you are installing in a server without GUI
sumo-gui

# setup virtual environment (Python 3.7 is required)
# or you can use conda environment if you like.
python3.7 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# install the dependencies
pip install smarts-0.3.7-py3-none-any.whl
pip install smarts-0.3.7-py3-none-any.whl[train]
pip install smarts-0.3.7-py3-none-any.whl[dev]

# download the public datasets from Codalab to ../dataset_public

# test that the sim works
python train_example/keeplane_example.py --scenario ../dataset_public/simple_loop/simpleloop_a

# if you recieve problems like "Exception: Could not open window." or "Cannot use SUMO"
# Go to FAQ Environment Config section.
```

## Docs

To look at the documentation call:

```bash
# Browser will attempt to open on localhost:8082
scl docs

# if you face problems, Go to FAQ Environment Config section
```


