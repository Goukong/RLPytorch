import torch.nn as nn
import torch
import gym
from Buffer import  Buffer
import numpy as np
from tensorboardX import SummaryWriter
import os
import math

class Net(nn.Module):
    def __init__(self,obs_dim,act_dim):
        super(Net,self).__init__()
        # self.fc1 = nn.Linear(obs_dim,32)
        # self.act1 = nn.ReLU(inplace=True)
        # self.fc2 = nn.Linear(32,32)
        # self.act2 = nn.ReLU(inplace=True)
        #
        # self.act_dim = act_dim
        # self.value = nn.Linear(32,1)
        # self.advantage = nn.Linear(32,act_dim)
        self.feature = nn.Sequential(nn.Linear(obs_dim,32),
                                     nn.ReLU())

        self.advantage_layer = nn.Sequential(nn.Linear(32,32),
                                             nn.ReLU(),
                                             nn.Linear(32,act_dim))
        self.value_layer = nn.Sequential(nn.Linear(32,32),
                                         nn.ReLU(),
                                         nn.Linear(32,1))


    def forward(self,x):
        feature = self.feature(x)
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)
        q = value + advantage - advantage.mean(dim=-1,keepdim=True)
        return q

class DuelingDQN():
    def __init__(self,obs_dim,act_dim,eps_decay=500,buf_size=5000,
                 mini_batch=256,epsilon=0.9,gamma=0.95,lr = 1e-2,delay_update=30):
        super(DuelingDQN,self).__init__()
        self.act_dim = act_dim

        self.q_eval = Net(obs_dim,act_dim)
        self.q_target = Net(obs_dim,act_dim)
        self.q_target.load_state_dict(self.q_eval.state_dict())
        self.q_target.eval()

        self.buffer = Buffer(buf_size,mini_batch)

        self.step = 0
        self.delay_upate = delay_update

        self.mini_batch = mini_batch
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_eval.parameters(),lr=lr)
        self.epsilon = epsilon
        self.gamma = gamma

        self.eps_start = epsilon
        self.eps_end = 0.05
        self.eps_decay = eps_decay
        self.step_done = 0

        self.path = 'run/DuelingDQN/'
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.logger = SummaryWriter(self.path,comment='DuelingDQN')

    def store(self,s,a,r,s_,d):
        self.buffer.store(s,a,r,s_,d)

    def choose_action(self,obs):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1.0 * self.step_done/self.eps_decay)
        self.step_done += 1
        if np.random.uniform() > eps_threshold:
            q_val = self.q_eval(torch.as_tensor(obs,dtype=torch.float32))
            act = q_val.argmax().item()
        else:
            act = np.random.choice(self.act_dim,1)[0]

        return act

    def test_choose_action(self,obs):
        q_val = self.q_eval(torch.as_tensor(obs, dtype=torch.float32))
        act = q_val.argmax.item()
        return act

    def get_ptr(self):
        return  self.buffer.ptr
    def learn(self):
        batch_data = self.buffer.get_data()
        # 将动作转换成LongTensor也就是int64
        at = torch.as_tensor(batch_data['act'], dtype=torch.int64).unsqueeze(1)
        q_eval = self.q_eval(batch_data['obs'])
        q_eval = q_eval.gather(1, at)

        rt = batch_data['rew']
        #先用eval(new)网络来求出当前的最大动作
        dump = self.q_eval(batch_data['next'])
        best_at = self.q_eval(batch_data['next']).argmax(1).unsqueeze(1)

        #然后用target(old)网络来求出该最佳动作对应的Q值
        qnext = self.q_target(batch_data['next'])
        #注意qnext要压缩一次，保持其是64，方便后面的乘法
        #否则，其shape为【64,1】，而终止项大小为【64】，这样会变成矩阵乘法,结果变成【64,64】大小
        qnext = qnext.gather(1,best_at).squeeze()
        q_target = rt+self.gamma * qnext*(1-batch_data['done'])
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
            self.q_target.load_state_dict(self.q_eval.state_dict())

        return loss
    def update(self):
        data = self.buffer.get_data()
        at = data['act'].long().unsqueeze(1)
        q_eval = self.q_eval(data['obs']).gather(1,at)
        rt = data['rew']

        best_at = self.q_eval(data['next']).argmax(1).unsqueeze(1)
        q_next = self.q_target(data['next']).gather(1,best_at).squeeze()
        q_target = rt + self.gamma*q_next*(1-data['done'])
        q_target = q_target.reshape(self.mini_batch,1)

        loss = self.loss_func(q_eval,q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step += 1
        if self.step % self.delay_upate == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())

        return loss

    def save(self,path=None,epoch=0,name = 'DuelingDQN'):
        dir = self.path+str(epoch)+name+'.pkl'
        torch.save(self.q_eval,dir)

    def load(self,epoch,name = 'DuelingDQN'):
        dir = self.path + str(epoch) + name+'.pkl'
        self.q_eval = torch.load(dir)



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

    dqn = DuelingDQN(obs_dim=STATE_DIM, act_dim=ACT_DIM, buf_size=BUFFER_SIZE,
                     mini_batch=MINI_BATCH, lr=LEARNING_RATE, epsilon=EPSILON, gamma=GAMA)

    for epoch in range(MAX_EPISODE):
        s = env.reset()
        rew = []
        while True:
            a = dqn.choose_action(s)
            s_, r, done, info = env.step(a)
            dqn.store(s, a, r, s_, done)
            rew.append(r)
            if dqn.get_ptr() > MINI_BATCH:

                loss = dqn.update()

                if done:
                    rew_sum = np.sum(rew)
                    print("EP:", epoch, "|EP_loss:", loss, "|EP_rew:", rew_sum)
                    dqn.logger.add_scalar('loss:', loss, global_step=epoch)
                    dqn.logger.add_scalar('sum_rew:', rew_sum, global_step=epoch)
                    break
            if done:
                break
            else:
                s = s_
        if epoch == MAX_EPISODE - 1:
            dqn.save()
            break
    env.close()

train()













