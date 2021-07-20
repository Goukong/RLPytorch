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

class DDPG():
    def __init__(self,env_fn,step_per_epoch=4000,epoch=100,batch_size=128,buffer_size=1000000,
                 pi_lr=1e-3,v_lr=1e-3,gamma=0.99,polyak=0.995,act_noise=0.1,max_ep_len=1000,
                 start_step=10000,update_after=1000,update_every=50):
        self.env = gym.make(env_fn)
        self.test_env = gym.make(env_fn)
        obs_space = self.env.observation_space
        act_space = self.env.action_space

        self.ac = core.ActorCritic(obs_space,act_space)
        self.ac_target = deepcopy(self.ac)

        self.obs_dim = obs_space.shape
        self.act_dim = act_space.shape[0]
        self.act_high = act_space.high[0]
        self.act_low = act_space.low[0]

        self.buffer = ReplayBuffer(obs_dim=self.obs_dim,act_dim=self.act_dim,buffer_size=buffer_size)

        self.pi_optimizer = torch.optim.Adam(self.ac.pi.parameters(),lr=pi_lr)
        self.q_optimizer = torch.optim.Adam(self.ac.q.parameters(),lr=v_lr)

        self.step_per_epoch = step_per_epoch
        self.epoch = epoch

        self.batch_size = batch_size
        self.gamma = gamma
        self.polyak = polyak
        self.act_noise = act_noise
        self.max_ep_len = max_ep_len
        self.start_step = start_step
        self.update_after = update_after
        self.update_every = update_every

        path = 'run/ddpg/'
        if not os.path.exists(path):
            os.makedirs(path)
        self.logger = SummaryWriter(path,comment='ddpg')

    def compute_q_loss(self,data):
        obs,act,rew,next_obs,done = data['obs'],data['act'],data['rew'],data['next_obs'],data['done']
        q_eval = self.ac.q(obs,act)
        with torch.no_grad():
            q_target = rew + self.gamma*(1-done)*self.ac_target.q(next_obs,self.ac_target.pi(next_obs))
        loss_q = ((q_eval-q_target)**2).mean()
        #返回q_loss 和 q值，q值用来判断网络的学习情况
        return loss_q,q_eval.detach().numpy()

    def compute_pi_loss(self,data):
        #data = self.buffer.get_batch(batch_size=self.batch_size)
        obs = data['obs']
        q_pi = self.ac.q(obs,self.ac.pi(obs))
        return -q_pi.mean()

    def update(self):
        data = self.buffer.get_batch(self.batch_size)

        self.q_optimizer.zero_grad()
        q_loss, q_val = self.compute_q_loss(data)
        q_loss.backward()
        self.q_optimizer.step()

        #q和pi在一个类里面，避免计算资源的浪费
        for p in self.ac.q.parameters():
            p.requires_grad=False

        self.pi_optimizer.zero_grad()
        pi_loss = self.compute_pi_loss(data)
        pi_loss.backward()
        self.pi_optimizer.step()

        for p in self.ac.q.parameters():
            p.requires_grad = True
        with torch.no_grad():
            #polyak 软更新
            for p,p_target in zip(self.ac.parameters(),self.ac_target.parameters()):
                p_target.data.mul_(self.polyak)
                p_target.data.add_((1-self.polyak)*p.data)

        return q_loss,pi_loss

    def get_action(self,obs,act_noise):
        act = self.ac.act(torch.as_tensor(obs,dtype=torch.float32))
        act += np.random.randn(self.act_dim) * act_noise
        return np.clip(act,self.act_low,self.act_high)

    def save(self,path='model/',name='ddpg',type='normal',epoch=0):
        if not os.path.exists(path):
            os.makedirs(path)
        save_dir = path + name + type + "__" +str(epoch)+".pkl"

        torch.save(self.ac,save_dir)

    def load(self,path='model/',name='ddpg',type='normal',epoch=0):
        if not os.path.exists(path):
            os.makedirs(path)
        load_dir = path + name + type + "__" +str(epoch)+".pkl"
        self.ac = torch.load(load_dir)
        self.ac_target = deepcopy(self.ac)

    def test(self,rander=False):
        test_num_episode = 10
        rewss = []
        for i in range(test_num_episode):
            obs,done,ep_rew,ep_len = self.test_env.reset(),False,0,0
            while not (done or (ep_len == self.max_ep_len)):
                if rander:
                    self.test_env.render()
                    time.sleep(0.1)
                act = self.get_action(obs,0)
                next_obs,rew,done,_ = self.test_env.step(act)
                obs = next_obs
                ep_rew += rew
                ep_len += 1
            #self.logger.add_scalar('test_rew',ep_rew/ep_len,i)
            rewss.append(ep_rew)
        return np.mean(rewss)

    def train(self):
        best_rew = -1e9
        total_step = self.step_per_epoch*self.epoch
        obs,ep_rew,ep_len,ep_done = self.env.reset(),0,0,0
        loss_q,loss_pi,total_rew = [],[],0
        for t in range(total_step):
            if t < self.start_step:
                act = self.env.action_space.sample()
            else:
                act = self.get_action(obs,self.act_noise)

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
                for _ in range(self.update_every):
                    l1,l2,log_pi,q_info = self.update()
                loss_q.append(l1.item())
                loss_pi.append(l2.item())
            if (t+1) % self.step_per_epoch == 0:
                epoch = (t+1) // self.step_per_epoch

                self.logger.add_scalar('epoch_rew_mean',total_rew/self.step_per_epoch,epoch)
                self.logger.add_scalar('epoch_loss_q',np.mean(loss_q),epoch)
                self.logger.add_scalar('epoch_loss_pi',np.mean(loss_pi),epoch)

                test_reward = self.test()
                if test_reward > best_rew:
                    self.save(type='best')
                    best_rew = test_reward

                print('epoch:{},rew_mean{},loss_q{},loss_pi{},test_rew{}'.format(epoch, total_rew,
                                                                      np.mean(loss_q), np.mean(loss_pi),test_reward))
                loss_q, loss_pi, total_rew = [], [], 0
if __name__ == '__main__':
    ddpg = DDPG('LunarLanderContinuous-v2')
    ddpg.load(type='best')
    ddpg.test(rander=True)






