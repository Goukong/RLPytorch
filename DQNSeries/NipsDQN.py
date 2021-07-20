import torch.nn as nn
import torch
import torch.optim as optim
import gym
import numpy as np
from Buffer import Buffer
import math
from tensorboardX import SummaryWriter
import os

class Net(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(state_dim,50)
        self.act1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(50,50)
        self.act2 = nn.ReLU(inplace=True)
        self.output = nn.Linear(50,action_dim)

    def forward(self,x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        #输出的是st下所有动作a所对应的Q值
        x = self.output(x)
        return x


EPSILON = 0.9
GAMA = 0.9
LEARNING_RATE = 0.001
BUFFER_SIZE = 1000
MINI_BATCH = 64
EPOCH = 100
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 500
steps_done = 0

class DQN():
    def __init__(self,state_dim,action_dim):
        self.model = Net(state_dim,action_dim)
        self.buffer = Buffer(BUFFER_SIZE,MINI_BATCH)
        self.optimizer = optim.Adam(self.model.parameters(),lr=LEARNING_RATE)
        self.loss_func = nn.MSELoss()

        self.path = 'run/NipsDQN/'
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.logger = SummaryWriter(self.path, comment='NipsDQN')

    def choose_action(self, s):
        global steps_done
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        #对于NIPSDQN来说：
        #在奖励函数设置正确的情况下
        #随机到的动作越多，收集的资料越多，效果越好
        #但还是会倒，就是不一定会收敛到最优
        #而如果是随着步数慢慢放弃探索，效果会很差
        #这就能看出与natrueDQN的差距
        if np.random.uniform() > eps_threshold:
            q_value = self.model(s)
            action = q_value.argmax().item()
        else:
            action = np.random.choice(ACTION_DIM, 1)[0]
        return action

    def test_choose_action(self,s):
        q_value = self.model(s)
        action = q_value.argmax().item()
        return  action

    def store_transision(self,s,a,r,s_,d):
        self.buffer.store(s,a,r,s_,d)

    def get_ptr(self):
        return self.buffer.ptr

    def learn(self):
        batch_data = self.buffer.get_data()
        # 将动作转换成LongTensor也就是int64
        at = torch.as_tensor(batch_data['act'],dtype=torch.int64).unsqueeze(1)
        # 然后根据这个动作，计算Q值，gather就是把每行选择的at结合到一起
        q_eval = self.model(batch_data['obs'])
        q_eval = q_eval.gather(1, at)

        rt = batch_data['rew']
        qnext = self.model(batch_data['next'])
        #20210406
        #done 这个信息很重要！
        #它能区分每个动作的奖励值，也就是说只有奖励值有差异的情况下，学习才是有可能的
        #除非进行连续性奖励函数的设计
        q_target = rt + GAMA * qnext.max(1)[0]*(1-batch_data['done'])
        q_target = q_target.unsqueeze(1)
        loss = self.loss_func(q_target,q_eval)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
    def save(self):
        torch.save(self.model,'cartpole_v0_NIPSDQN.pkl')

    def load(self,path):
        self.model = torch.load(path)




env = gym.make("CartPole-v0")
ACTION_DIM = env.action_space.n
STATE_DIM = env.observation_space.shape[0]

def train():
    model = DQN(STATE_DIM,ACTION_DIM)
    loss_dump = []
    rew_dump = []
    for i in range(EPOCH):
        s = env.reset()
        while True:
            a = model.choose_action(torch.as_tensor(s,dtype=torch.float32))
            s_,r,done,info = env.step(a)
            #这种奖励函数的设置，使向左或向右产生差别，是一定要区分的！
            #对于DQN这种基于值函数的分析，每一步的区别很重要！
            # x, x_dot, theta, theta_dot = s_
            # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8  # env.x_threshold=2.4,在cartpole.py中有写
            # r2 = (env.theta_threshold_radians - abs(
            #     theta)) / env.theta_threshold_radians - 0.5  # env.theta_threshold_radians=12度，同样在cartpole.py中有写
            # r = r1 + r2

            rew_dump.append(r)
            #20210406
            #加入done信息后，就产生了差别
            model.store_transision(s, a, r, s_,done)
            if model.get_ptr() > MINI_BATCH:
                loss = model.learn()
                loss_dump.append(loss.item())
                if done:
                    print("EP:",i,"|EP_loss:",np.mean(loss_dump),"|EP_rew:",np.sum(rew_dump))
                    model.logger.add_scalar('loss:',loss,global_step=i)
                    model.logger.add_scalar('sum_rew:', np.sum(rew_dump), global_step=i)
                    loss_dump = []
                    rew_dump = []
            if done:
                break
            else:
                s = s_
        if i == EPOCH-1:
            model.save()
            break
    env.close()
def test():
    import time
    t = 0
    model = DQN(STATE_DIM,ACTION_DIM)
    model.load('cartpole_v0_NIPSDQN.pkl')

    s = env.reset()

    while True:
        t += 1
        env.render()
        print(t)
        time.sleep(0.1)
        a = model.test_choose_action(torch.as_tensor(s,dtype=torch.float32))
        s_, r, done, info = env.step(a)
        if done:
            break
        else:
            s = s_
    env.close()
train()
test()


