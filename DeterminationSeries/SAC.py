import itertools

import torch
import numpy as np
from tensorboardX import SummaryWriter
import core
import gym
from copy import deepcopy
import os
import time

class ReplayBuffer():
    def __init__(self,obs_dim,act_dim,buffer_size):

        self.obs_buf = np.zeros(core.combine_shape(buffer_size,obs_dim),dtype=np.float32)
        self.next_obs = np.zeros(core.combine_shape(buffer_size,obs_dim),dtype=np.float32)
        self.act = np.zeros(core.combine_shape(buffer_size,act_dim),dtype=np.float32)
        self.rew = np.zeros(buffer_size,dtype=np.float32)
        self.done = np.zeros(buffer_size,dtype=np.float32)
        self.ptr,self.cur_size,self.max_size = 0,0,buffer_size

    def store(self,obs,act,rew,next_obs,done):
        self.obs_buf[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done
        self.ptr = (self.ptr+1)%self.max_size
        self.cur_size += 1
        self.size = min(self.cur_size+1,self.max_size)

    def get_batch(self,batch_size=32):
        index = np.random.randint(0,self.size,batch_size)
        batch = dict(
            obs = self.obs_buf[index],
            act = self.act[index],
            rew = self.rew[index],
            next_obs = self.next_obs[index],
            done = self.done[index]
        )
        return {k:torch.as_tensor(v,dtype=torch.float32) for k,v in batch.items()}

class SAC():
    def __init__(self,env_fn,step_per_epoch=4000,epoch=100,batch_size=128,buffer_size=1000000,
                 pi_lr=1e-3,v_lr=1e-3,gamma=0.99,polyak=0.995,max_ep_len=1000,alpha=0.2,
                 start_step=10000,update_after=1000,update_every=50):
        super(SAC, self).__init__()
        self.env = gym.make(env_fn)
        self.test_env = gym.make(env_fn)
        obs_space = self.env.observation_space
        act_space = self.env.action_space

        self.ac = core.SACActorCritic(obs_space, act_space)
        self.ac_target = deepcopy(self.ac)
        for p in self.ac_target.parameters():
            p.requires_grad = False

        #状态可以是多维度的，动作一般都是单维的
        self.obs_dim = obs_space.shape
        self.act_dim = act_space.shape[0]
        self.act_high = act_space.high[0]
        self.act_low = act_space.low[0]

        self.buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, buffer_size=buffer_size)

        self.pi_optimizer = torch.optim.Adam(self.ac.pi.parameters(), lr=pi_lr)

        #
        self.q_params = itertools.chain(self.ac.q1.parameters(),self.ac.q2.parameters())
        self.q_optimizer = torch.optim.Adam(self.q_params, lr=v_lr)

        self.step_per_epoch = step_per_epoch
        self.epoch = epoch

        self.batch_size = batch_size
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.max_ep_len = max_ep_len
        self.start_step = start_step
        self.update_after = update_after
        self.update_every = update_every

        path = 'run/sac/'
        if not os.path.exists(path):
            os.makedirs(path)
        self.logger = SummaryWriter(path, comment='sac')

    def save(self,path='model/',name='sac',type='normal',epoch=0):
        if not os.path.exists(path):
            os.makedirs(path)
        save_dir = path + name + type + "__" +str(epoch)+".pkl"

        torch.save(self.ac,save_dir)

    def load(self,path='model/',name='sac',type='normal',epoch=0):
        if not os.path.exists(path):
            os.makedirs(path)
        load_dir = path + name + type + "__" +str(epoch)+".pkl"
        self.ac = torch.load(load_dir)
        self.ac_target = deepcopy(self.ac)

    def compute_q_loss(self,data):
        obs,act,rew,next_obs,done = data['obs'],data['act'],data['rew'],data['next_obs'],data['done']
        q_eval1 = self.ac.q1(obs,act)
        q_eval2 = self.ac.q2(obs,act)

        with torch.no_grad():
            #根据公式计算Q_target
            next_act,log_pi = self.ac.pi(next_obs)
            q_next_target1 = self.ac_target.q1(next_obs,next_act)
            q_next_target2 = self.ac_target.q2(next_obs,next_act)
            q_next_target = torch.min(q_next_target1,q_next_target2)
            q_target = rew + self.gamma*(1-done)*(q_next_target - self.alpha * log_pi)

        loss1 = ((q_eval1-q_target)**2).mean()
        loss2 = ((q_eval2-q_target)**2).mean()
        loss = loss1 + loss2

        q_info = dict(q1_vals = q_eval1.detach().numpy(),
                      q2_vals = q_eval2.detach().numpy())

        return loss,q_info

    def compute_pi_loss(self,data):
        obs = data['obs']
        act,log_pi = self.ac.pi(obs)
        #累计奖励就是用当前的状态动作来进行估计的q值
        q1 = self.ac.q1(obs,act)
        q2 = self.ac.q2(obs,act)
        q = torch.min(q1,q2)
        loss_pi = (self.alpha*log_pi-q).mean()

        return loss_pi,log_pi.detach().numpy()

    def get_action(self,obs,deterministic=False):
        action = self.ac.act(torch.as_tensor(obs,dtype=torch.float32),deterministic)
        return action

    def update(self):
        data = self.buffer.get_batch(self.batch_size)

        loss_q,q_info = self.compute_q_loss(data)
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        for p in self.q_params:
            p.requires_grad = False

        loss_pi,log_pi = self.compute_pi_loss(data)
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()

        for p in self.q_params:
            p.requires_grad = True

        with torch.no_grad():
            for p,p_target in zip(self.ac.parameters(),self.ac_target.parameters()):
                p_target.data.mul_(self.polyak)
                p_target.data.add_((1-self.polyak)*p.data)

        return loss_q,loss_pi,log_pi,q_info

    def train(self):
        best_rew = -1e9
        total_step = self.step_per_epoch*self.epoch
        obs,ep_rew,ep_len,ep_done = self.env.reset(),0,0,0
        loss_q,loss_pi,total_rew = [],[],0
        log_pi_list,q1val,q2val = [],[],[]
        for t in range(total_step):
            if t < self.start_step:
                act = self.env.action_space.sample()
            else:
                act = self.get_action(obs,deterministic=False)

            next_obs,rew,done,_ = self.env.step(act)

            ep_rew += rew
            total_rew += rew

            ep_len += 1
            done = False if ep_len==self.max_ep_len else done
            self.buffer.store(obs,act,rew,next_obs,done)
            obs = next_obs

            if done or (ep_len == self.max_ep_len):
                self.logger.add_scalar('done_rew_mean',ep_rew/ep_len,global_step=ep_done)
                ep_done += 1
                obs,ep_rew,ep_len = self.env.reset(),0,0
            if t >= self.update_after and t % self.update_every == 0:
                l1, l2, log_pi, q_info = 0, 0, 0, None
                for _ in range(self.update_every):
                    l1,l2,log_pi,q_info = self.update()
                loss_q.append(l1.item())
                loss_pi.append(l2.item())
                log_pi_list.append(log_pi)
                q1val.append(q_info['q1_vals'])
                q2val.append(q_info['q2_vals'])
            if (t+1) % self.step_per_epoch == 0:
                epoch = (t+1) // self.step_per_epoch

                self.logger.add_scalar('epoch_rew',total_rew,epoch)
                self.logger.add_scalar('epoch_loss_q',np.mean(loss_q),epoch)
                self.logger.add_scalar('epoch_loss_pi',np.mean(loss_pi),epoch)
                self.logger.add_scalar('entropy',np.mean(log_pi_list),epoch)
                self.logger.add_scalar('q1:',np.mean(q1val),epoch)
                self.logger.add_scalar('q2:',np.mean(q2val),epoch)

                test_reward = self.test()
                if test_reward > best_rew:
                    self.save(type='best')
                    best_rew = test_reward

                print('epoch:{},rew_mean{},loss_q{},loss_pi{},test_rew{}'.format(epoch, total_rew,
                                                                      np.mean(loss_q), np.mean(loss_pi),test_reward))
                loss_q, loss_pi, total_rew = [], [], 0

    def test(self,rander=False):
        ep_ret = 0
        for j in range(10):
            obs,done,ep_ret,ep_len = self.test_env.reset(),False,0,0
            while not (done or ep_len == self.max_ep_len):
                if rander:
                    self.test_env.render()
                    time.sleep(0.01)
                obs,rew,done,_ = self.test_env.step(self.get_action(obs,True))
                ep_ret += rew
                ep_len += 1
            self.logger.add_scalar('test_rew:',ep_ret,j)
            print(ep_ret)
        print(ep_ret)
        return ep_ret

if __name__ == '__main__':
    sac = SAC('LunarLanderContinuous-v2')
    #sac.train()
    sac.load(type='best')
    sac.test(rander=True)



