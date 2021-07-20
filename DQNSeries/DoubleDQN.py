#基本上和natruedqn相同
#只有目标网络的用法发生了变化

import torch.nn as nn
import torch
import gym
from Buffer import Buffer
import numpy as np
from tensorboardX import SummaryWriter
import math
import os
class Net(nn.Module):
    def __init__(self,state_dim,act_dim):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(state_dim,32)
        self.act1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(32,32)
        self.act2 = nn.ReLU(inplace=True)
        self.output = nn.Linear(32,act_dim)

    def forward(self,x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.output(x)
        return x


class DoubleDQN():
    def __init__(self,state_dim,act_dim,buffer_size=4000,mini_batch=128,learning_rate = 0.01,epsilon=0.9,gama = 0.95):
        super(DoubleDQN, self).__init__()
        self.act_dim = act_dim
        self.qeval = Net(state_dim,act_dim)
        self.qtarget = Net(state_dim,act_dim)
        self.qtarget.load_state_dict(self.qeval.state_dict())
        self.buffer = Buffer(buffer_size,mini_batch)
        self.step = 0
        self.mini_batch = mini_batch
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.qeval.parameters(),lr = learning_rate)
        self.epsilon = epsilon
        self.gama = gama

        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 500
        self.steps_done = 0

        self.path = 'run/DoubleDQN/'
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.logger = SummaryWriter(self.path,comment='DoubleDQN')

    def store(self,s,a,r,s_,d):
        self.buffer.store(s,a,r,s_,d)

    def choose_action(self, s):
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        #self.logger.add_scalar('eps:',eps_threshold,self.steps_done)
        self.steps_done += 1
        if np.random.uniform() > eps_threshold:
            q_value = self.qeval(torch.as_tensor(s,dtype=torch.float32))
            action = q_value.argmax().item()
        else:
            action = np.random.choice(self.act_dim, 1)[0]
        return action

    def test_choose_action(self, s):
        q_value = self.qeval(torch.as_tensor(s,dtype=torch.float32))
        action = q_value.argmax().item()
        return action

    def get_ptr(self):
        return self.buffer.ptr

    def learn(self):
        batch_data = self.buffer.get_data()
        # 将动作转换成LongTensor也就是int64
        at = torch.as_tensor(batch_data['act'], dtype=torch.int64).unsqueeze(1)
        q_eval = self.qeval(batch_data['obs'])
        q_eval = q_eval.gather(1, at)

        rt = batch_data['rew']
        #先用eval(new)网络来求出当前的最大动作
        dump = self.qeval(batch_data['next'])
        best_at = self.qeval(batch_data['next']).argmax(1).unsqueeze(1)

        #然后用target(old)网络来求出该最佳动作对应的Q值
        qnext = self.qtarget(batch_data['next'])
        #注意qnext要压缩一次，保持其是64，方便后面的乘法
        #否则，其shape为【64,1】，而终止项大小为【64】，这样会变成矩阵乘法,结果变成【64,64】大小
        qnext = qnext.gather(1,best_at).squeeze()
        q_target = rt+self.gama * qnext*(1-batch_data['done'])
        #q_target要reshape和q_eval一样的大小，这样计算的loss才准确！
        q_target = q_target.reshape(self.mini_batch,1)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #延迟更新
        self.step += 1
        #延迟更新快一点，效果就会不稳定
        #20 血崩了
        #30 还可以
        #50 比较稳
        if self.step % 30 == 0:
            self.qtarget.load_state_dict(self.qeval.state_dict())

        return loss
    def save(self):
        torch.save(self.qeval,'cartpole_v0_DoubleDQN.pkl')

    def load(self,path):
        self.qeval = torch.load(path)

def train():
    env = gym.make('CartPole-v0')
    STATE_DIM = env.observation_space.shape[0]
    ACT_DIM = env.action_space.n

    MAX_EPISODE = 100
    EPSILON = 0.9
    GAMA = 0.9
    LEARNING_RATE = 0.001
    BUFFER_SIZE = 1000
    MINI_BATCH = 64

    dqn = DoubleDQN(STATE_DIM,ACT_DIM,BUFFER_SIZE,MINI_BATCH,LEARNING_RATE,EPSILON,GAMA)

    for epoch in range(MAX_EPISODE):
        s = env.reset()
        rew = []
        while True:
            a = dqn.choose_action(s)
            s_,r,done,info = env.step(a)
            dqn.store(s,a,r,s_,done)
            rew.append(r)
            if dqn.get_ptr() > MINI_BATCH:
                loss = dqn.learn()
                if done:
                    rew_sum = np.sum(rew)
                    print("EP:",epoch,"|EP_loss:",loss,"|EP_rew:",rew_sum)
                    dqn.logger.add_scalar('loss:',loss,global_step=epoch)
                    dqn.logger.add_scalar('sum_rew:',rew_sum,global_step=epoch)
                    break
            if done:
                break
            else:
                s = s_
        if epoch == MAX_EPISODE-1:
            dqn.save()
            break
    env.close()
def test():
    import time
    t = 0

    env = gym.make('CartPole-v0')
    STATE_DIM = env.observation_space.shape[0]
    ACT_DIM = env.action_space.n
    model = DoubleDQN(STATE_DIM,ACT_DIM)
    model.load('cartpole_v0_DoubleDQN.pkl')

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