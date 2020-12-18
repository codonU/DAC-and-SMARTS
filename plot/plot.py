import numpy as np
import matplotlib.pyplot as plt
import os
import re

path = os.getcwd() + '/plot/csv/'
env = ['dm/', 'openai/']

dm = {
    'cartpole': [],
    'reacher': [],
    'cheetah': [],
    'fish': [],
    'walker-1': [],
    'walker-2': []
}
openai = {
    'HalfCheetah': [],
    'Walker2d': [],
    'Hopper': [],
    'Swimmer': []
}

csv_dm = os.listdir(path + env[0])
key_dm = dm.keys()
for csv in csv_dm:
    for key in key_dm:
        if key in csv:
            dm[key].append(csv)
for key in key_dm:
    print(key, len(dm[key]))

csv_openai = os.listdir(path + env[1])
key_openai = openai.keys()
for csv in csv_openai:
    for key in key_openai:
        if key in csv:
            openai[key].append(csv)
for key in key_openai:
    print(key, len(openai[key]))


def plot(env, task, l_smooth=0.6, dm_PPO=False):
    if env == 'dm':
        data = dm
    elif env == 'openai':
        data = openai
    for d in data[task]:
        line = np.loadtxt(path + env + '/' + d, delimiter=',', skiprows=1, unpack=True)[1:]
        name = re.search(r'remark_\w*', d).group().replace('remark_', '')
        name = name_change(env, name)
        if name == 'PPO' and dm_PPO:
            pass
        else:
            plt.plot(line[0], smooth(line[1], l_smooth), label=name)
    plt.title(env + '-' + task)
    plt.legend()
    plt.show()
    plt.ion()


def smooth(data, weight=0.6):
    smu = np.zeros_like(data)
    last = data[0]
    for i in range(len(data)):
        smu[i] = last * weight + (1 - weight) * data[i]
        last = smu[i]
    return smu


def name_change(env, name):
    if env == 'dm':
        if name == 'AHP':
            return 'AHP + PPO'
        if name == 'ASC':
            return 'DAC + PPO'
    return name
# a = np.loadtxt(f, str, delimiter=',', skiprows=1, unpack=True)
# print(a)

# for key in dm.keys():
#     plot('dm', key, l_smooth=0.9)
for key in openai.keys():
    plot('openai', key, l_smooth=0.9)