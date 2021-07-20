import numpy as np
import torch
class Buffer():
    def __init__(self,buffer_size = 5000,mini_batch = 256):
        super(Buffer,self).__init__()
        self.buffer_size = buffer_size
        self.obs_buf = []
        self.act_buf = []
        self.next_buf = []
        self.rew_buf = []
        self.done_buf = []
        self.mini_batch = mini_batch
        self.ptr = 0
    #data = [st,at,rt,st+1,d]
    def store(self,s,a,r,st,d):
        if self.ptr < self.buffer_size:
            self.obs_buf.append(s)
            self.act_buf.append(a)
            self.next_buf.append(st)
            self.rew_buf.append(r)
            self.done_buf.append(d)
        else:
            #旧的覆盖新的
            self.obs_buf[self.ptr%self.buffer_size] = s
            self.act_buf[self.ptr % self.buffer_size] = a
            self.next_buf[self.ptr % self.buffer_size] = st
            self.rew_buf[self.ptr % self.buffer_size] = r
            self.done_buf[self.ptr % self.buffer_size] = d
        self.ptr += 1
        if self.ptr == self.buffer_size:
            self.ptr = 0
    #采样mini-batch 用来训练
    def get_data(self):
        assert self.ptr > self.mini_batch,"当前回放池的数据还不够训练"
        batch_index = np.random.choice(min(self.ptr,self.buffer_size),self.mini_batch,replace = False)
        batch_obs = np.array(self.obs_buf)[batch_index]
        batch_act = np.array(self.act_buf)[batch_index]
        batch_next = np.array(self.next_buf)[batch_index]
        batch_rew = np.array(self.rew_buf)[batch_index]
        batch_done = np.array(self.done_buf)[batch_index]
        batch_data = dict(obs=batch_obs,act=batch_act,next=batch_next,rew=batch_rew,done=batch_done)
        #随机采样完的数据，还要转换成tensor才能返回
        return {k:torch.as_tensor(v,dtype=torch.float32) for k,v in batch_data.items()}