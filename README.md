# DAC：some option algorithms

This branch is the code for the paper

*DAC: The Double Actor-Critic Architecture for Learning Options* \
Shangtong Zhang, Shimon Whiteson (NeurIPS 2019)

    .
    ├── Dockerfile                                      # Dependencies
    ├── requirements.txt                                # Dependencies
    ├── template_jobs.py                                
    |   ├── batch_mujoco                                # Start Mujoco experiments 
    |   ├── batch_dm                                    # Start DMControl experiments 
    |   ├── a_squared_c_ppo_continuous                  # Entrance of DAC+PPO
    ├── deep_rl/agent/ASquaredC_PPO_agent.py            # Implementation of DAC+PPO 
    ├── deep_rl/agent/ASquaredC_A2C_agent.py            # Implementation of DAC+A2C 
    ├── deep_rl/agent/AHP_PPO_agent.py                  # Implementation of AHP+PPO 
    ├── deep_rl/agent/IOPG_agent.py                     # Implementation of IOPG 
    ├── deep_rl/agent/OC_agent.py                       # Implementation of OC 
    ├── deep_rl/agent/PPOC_agent.py                     # Implementation of PPOC 
    ├── deep_rl/component/cheetah_backward.py           # Cheetah-Backward 
    ├── deep_rl/component/walker_ex.py                  # Walker-Backward/Squat 
    ├── deep_rl/component/fish_downleft.py              # Fish-Downleft 
    └── plot_paper.py                                   # Plotting

> I can send the data for plotting via email upon request.

> This branch is based on the DeepRL codebase and is left unchanged after I completed the paper. Algorithm implementations not used in the paper may be broken and should never be used. It may take extra effort if you want to rebase/merge the master branch.



# SMARTS: auto-drive simulation

参考starter_kit里官方的README.md



# 运行

## 运行本库

运行的主文件为`template_SMARTS.py`

## log

### 2020.12.26

目前能够在SMARTS的各种地图上跑DAC库中的各个算法

**目前的代码逻辑**：

DAC option库 与SMARTS库各自有训练器，但是由于DAC中实现的各算法，环境是在agent之中，更不容易改动，所以参照SMARTS中`keep_lane`的实例单独创建环境。原来创建环境只需要环境名称，现在需要额外的agent接受环境变量，agent_id等参数，所以创建了字典代替原来的单独的字符串环境名，agent_spec选择了`continuous.py`中默认的agent_spec。然后在`env.py`中新写了一个适用于SMARTS环境的包装类`SMARTSWrapper`，将返回的observation、action、done、info都对其特有的agent_id获得，返回符合gym格式的变量。

但是有以下明显问题未处理：

1. logger中：暂时只对SMARTS环境在logger中直接改为"SMARTS"，暂时不能区分道路与算法
2. SMARTS道路：目前只参考`keep_lane.py`实现了单道路，只能通过config的初始化切换环境，暂时不能自动地跑过所有道路
3. 超参：各种算法的超参都需要重新调整





# 两个库的环境配置



## DAC环境

### 1. 安装requirement.txt
#### 1.1 安装torch=0.4.0有问题
DAC原环境要求python=3.6，SMARTS必须为3.7，在conda创建3.7的python环境后找不到匹配的0.4.0的torch
目前直接安装最新版torch，`conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`，安装后为torch1.7
#### 1.2 gym==0.10.8 check
#### 1.3 tensorflow==1.12.0有问题
改为1.15.0
#### 1.4 deepmind dm_control
dm_control的安装`git+git://github.com/deepmind/dm_control.git@1038345fa347165b7e29fa677824b2e9ee87762f` 需要mujoco的文件夹名为mujoco200最初的`mujoco200_linux`但是之前重装的时候名字改为了`mujoco200`
重新把`mujoco200_linum`解压了一下，check
#### 1.5 其他
其他都正常安装
opencv-python和roboschool直接安装，没有版本控制
#### 1.6 baseline
使用dockerfile内`pip install git+git://github.com/openai/baselines.git@8e56dd#egg=baselines`直接可用

### 2. 安装mujoco-py
requirement里的1.50的mujoco-py对应的版本是150的mujoco
直接装最新的即可
装最新的mujoco-py中间有报错 `AttributeError: module 'enum' has no attribute 'IntFlag'`
https://blog.csdn.net/weixin_41010198/article/details/87255393

卸载enum34后直接安装mujoco-py成功

### 3. 安装SMARTS
还是按照setup.md里第二部分直接安装就好

### 4. gym版本的问题
DAC库里gym是0.10.8，配置完DAC环境，再安装完SMARTS，运行SMARTS的测试示例会遇到
`gym 中的 make() got an unexpected keyword argument 'scenarios'`

重新安装最新的gym后，SMARTS的问题解决，但是DAC的代码不能运行了

### 5. 对DAC作改动
注释了DAC中envs.py中的import roboschool

## SMARTS环境

按照starter_kit中setup.md  `Setup from scratch`部分

