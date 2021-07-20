import core
import numpy as np
import torch

class Buffer():
    #与DQN系列不同，这里的buffer是直接申请了size大小的空间，然后通过ptr进行控制填充
    def __init__(self,size,obs_dim,act_dim,gamma = 0.99,lam = 0.95):
        super(Buffer,self).__init__()
        #st,at用于计算log(\pi(at|st))
        self.obs_buf = np.zeros(core.combine_shape(size,obs_dim),dtype=np.float32)
        self.act_buf = np.zeros(core.combine_shape(size,act_dim),dtype=np.float32)
        self.next_obs_buf = np.zeros(core.combine_shape(size,obs_dim),dtype=np.float32)
        self.logp_buf = np.zeros(size,dtype=np.float32)

        #rt 用于计算reward-to-go
        self.rew_buf = np.zeros(size,dtype=np.float32)
        self.rtg_buf = np.zeros(size,dtype=np.float32)

        #vt 即第t步的时间差分项，用于计算GAE
        self.val_buf = np.zeros(size,dtype=np.float32)
        self.adv_buf = np.zeros(size,dtype=np.float32)
        self.gamma = gamma
        self.lam = lam

        #ptr_start_idx 到 ptr 就是这轮迭代所收集到的数据
        self.ptr,self.ptr_start_idx,self.max_size = 0,0,size

    def store(self,obs,act,rew,val,logp,next_obs):
        assert self.ptr < self.max_size #防止溢出
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.next_obs_buf[self.ptr] = next_obs
        self.ptr += 1

    #路径结束，需要计算GAE和rtg了
    #记录last_val的原因是路径的结束可能不是因为轨迹真正结束才停止的
    def finish_path(self,last_val = 0):
        #首先划分出，当前轨迹的区间
        path_slice = slice(self.ptr_start_idx,self.ptr)

        rews = np.append(self.rew_buf[path_slice],last_val)
        vals = np.append(self.val_buf[path_slice],last_val)

        #GAE(gamma,lam) = (gamma*lam)^t' * (delta_(t+t'))
        #delta_t = rt + val_t+1 - val_t
        delta = rews[:-1] + self.gamma * vals[1:] - vals[:-1]

        self.adv_buf[path_slice] = core.discount_cumsum(delta,self.gamma*self.lam)
        #!!!!!!!!!!!这里要少取一项，因为最后一项是额外加的
        self.rtg_buf[path_slice] = core.discount_cumsum(rews,self.gamma)[:-1]

        #将ptr_start更新为ptr，方便下次的存储
        self.ptr_start_idx = self.ptr

    #取出全部数据进行训练
    def get(self):
        #只有buffer满的时候，才取
        #!也就是说是靠存储的结构来区分同策略还是异策略的
        assert self.ptr == self.max_size
        self.ptr,self.ptr_start_idx = 0,0

        def get_statistics(x):
            mean = np.mean(x)
            std = np.std(x)
            return mean,std

        #对GAE进行正则化，能使其更快适应新的数据分布
        adv_mean,adv_std = get_statistics(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        data = dict(obs = self.obs_buf, act = self.act_buf, ret = self.rtg_buf,
                    adv = self.adv_buf, logp = self.logp_buf, rew = self.rew_buf,
                    next_obs = self.next_obs_buf)
        return {k:torch.as_tensor(v,dtype=torch.float32) for k,v in data.items()}

