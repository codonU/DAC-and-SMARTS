Update in 2020.08.21, sync with codalab FAQ page

# FAQ

## Participate

### 1. Where can I get the competition package?
After registered the CodaLab account, go to the participate page and send a request

### 2. How long will the participation request be approved?
In 6 hours，and the result will be sent to your registered email, pls check it in time.

### 3. What's the DDL of grouping?
Before the end of the first stage.

### 4. What's the max number of members in a team?
Up to 3 players is allowed, see terms-6.

## Environment Config

#### 1. When get into the docker, Fatal server error: (EE) Server is already active for display 1
Just ignore this error. This is to make sure the Xorg is running.

### 2. Can smarts simulator run on Windows?
Our simulator was developed for ubuntu (>=16.04) and macOS(>=10.15), but not suitable for WSL1 and WSL2. To install it on the Windows system, some prerequisites need to be met: (1) system version >= 10; (2) install it via docker (>=19.03.7).


### 3. Exception: Could not open window.

If you are running on a computer with GUI interface and occur this problem, **do not** try the solution below and try to use docker 
solution or contact us.

Otherwise if you are running on a server without GUI, you can try the following instructions to solve it.

```bash
    # set DISPLAY 
    vim ~/.bashrc
    # write following command into bashrc
    export DISPLAY=":1"
    # refresh
    source ~/.bashrc

    # set xorg server
    sudo wget -O /etc/X11/xorg.conf http://xpra.org/xorg.conf
    sudo /usr/bin/Xorg -noreset +extension GLX +extension RANDR +extension RENDER -logfile ./xdummy.log -config /etc/X11/xorg.conf $DISPLAY &
```

### 3. Cannot use sumo. You can export sumo path to bashrc manually

```bash
    # set SUMO HOME
    vim ~/.bashrc
    # write following command into bashrc
    # for ubuntu
    export SUMO_HOME="/usr/share/sumo"
    # for macos
    export SUMO_HOME="/usr/local/opt/sumo/share/sumo"
    # refresh
    source ~/.bashrc
```

### 4. When I run scl docs, it returns error. 

```bash
    Error: No docs found, try running:
    make docs
```

The reason is that you install the smarts package without any virtual environment likes virtualenv or conda (in other words, virtualenv and conda are recommended). It will return error.


That means scl cannot find the `smarts_docs` in `/usr/`, instead in `/usr/local/`. You can fix this error with soft link: `ln -s /usr/local/smarts_docs /usr/smarts_docs`, then `scl docs` will work successfully !
 
### 5. Core dumped when build scenaios.
  Since scenario building is parallel, this error means you do not have enough resources to do building cocurrently. Try to build scenario one by one.
  
## Envision
To make envision work, following requirements should be satisfied.
1. `supervisord` or `scl` command is running
2. the path in supervisor.conf or `-s` in `scl` should be pointed to dataset correctly

### 1. Adress already in use

Envision will use one port 8081, this error shows you have another program using this port, just kill this process or restart computer.

### 2. Cars are rendered, but roads are not rendered properly

`supervisord` default assume that starter-kit and dataset_public are in the same dir level, if not, modified the default path in supervisord.conf.

### 3. Open envision, but no roads or cars is rendered

If localhost:8081 can not be accessed, make sure you have open envision port by `supervisord`  or `scl` command.

If  localhost:8081 can be accessed but no rendered cars and roads, make sure `headless` mode is not set and `scl`  scenario path is correct.

If  you still have problem, raise that in the wechat group or Forums.

### 4. see envision on a remote server
use ssh port forwarding like
```bash
    ssh ssh -L8081:localhost:8081 -L8082:localhost:8082 -L6006:localhost:6006 username@server_ip 
```

## Submission

###  1. import agent error

The submission zip file requires just zip the outer directory like `submission_example` and also in this directory you must have a file named  `agent.py`.

###  2. import other modules error

This means you use some modules that are not installed in the evulation environment, contact us in the wechat group or forums.

## Training

### 1. no valid scenario found

build the scenario by `scl scenario build-all ../dataset_public`
