import numpy as np
import torch.nn as nn
import torch
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from gym.spaces import Box,Discrete

#该函数是为了应对状态或者动作是多维的情况，即dim不是常数是一个列表的情况
def combine_shape(length,shape = None):
    if shape is None:
        return (length,)
    return (length,shape) if np.isscalar(shape) else (length,*shape)

#计算GAE和rtg的平民写法
def discount_cumsum(x,discount):
    y = np.zeros_like(x)
    for i in reversed(range(len(x))):
        y[i] = x[i] if i == len(x)-1 else x[i] + discount*x[i+1]
    return y

#生成全连接网络
#sizes最后一维是动作的维度，因此到最后一个链接时必选用Identity
def mlp(sizes,activation,output_activation = nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i],sizes[i+1]),act()]
    return nn.Sequential(*layers)

#!!!开始构建网络的基础架构
class Actor(nn.Module):
    #随机性策略，给定一个obs，输出的是一个动作概率的分布
    def _distribution(self,obs):
        raise NotImplementedError
    #用于计算log的值
    def _log_prob_from_distribution(self,pi,act):
        raise NotImplementedError
    def forward(self,obs,act = None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi,act)
        return pi,logp_a
#连续性状态空间
#没太理解，先不考虑
class MLPGaussianActor(Actor):
    def __init__(self,obs_dim,act_dim,hidden_sizes,activation):
        super(MLPGaussianActor, self).__init__()
        #不懂什么意思，这样计算方差的目的在哪里？
        log_std = -0.5 * np.ones(act_dim,dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))
        #用于输出均值
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim],activation)

    def _distribution(self,obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu,std)
    #sum是为了让输出变成(batch,1)的形式。。。
    def _log_prob_from_distribution(self,pi,act):
        return pi.log_prob(act).sum(axis = -1)

class MLPCategoricalActor(Actor):
    def __init__(self,obs_dim,act_dim,hidden_sizes,activation):
        super(MLPCategoricalActor, self).__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim],activation)
    #学的是分布参数，构成分布后形成策略pi
    def _distribution(self,obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)
    def _log_prob_from_distribution(self,pi,act):
        return pi.log_prob(act)

class MLPCritic(nn.Module):
    def __init__(self,obs_dim,hidden_sizes,activation):
        super(MLPCritic, self).__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1],activation)
    def forward(self,obs):
        return torch.squeeze(self.v_net(obs),-1)

class MLPActorCritic(nn.Module):
    def __init__(self,obs_space,act_space,hidden_sizes=(64,64),activation = nn.Tanh):
        super(MLPActorCritic, self).__init__()
        obs_dim = obs_space.shape[0]

        if isinstance(act_space,Box):
            self.pi = MLPGaussianActor(obs_dim,act_space.shape[0],hidden_sizes,activation)
        elif isinstance(act_space,Discrete):
            self.pi = MLPCategoricalActor(obs_dim,act_space.n,hidden_sizes,activation)

        self.v = MLPCritic(obs_dim,hidden_sizes,activation)

    #用于根据obs生成相关动作
    def step(self,obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi,a)
            v = self.v(obs)
        return a.numpy(),v.numpy(),logp_a.numpy()

    def act(self,obs):
        return self.step(obs)[0]

class ICM(nn.Module):
    def __init__(self,obs_dim,act_dim,feature_dim=32,hidden_sizes=(64,64),activation=nn.Tanh):
        super(ICM , self).__init__()

        #encoder
        self.feature = mlp([obs_dim]+list(hidden_sizes)+[feature_dim],activation)

        self.forward_model = mlp([feature_dim+act_dim]+list(hidden_sizes)+[feature_dim],activation)

        self.inverse_model = mlp([feature_dim*2]+list(hidden_sizes)+[act_dim],activation)

    def forward(self,obs,act,next_obs):
        phi_st = self.feature(obs)
        phi_st1 = self.feature(next_obs)
        phi_cat = torch.cat([phi_st,phi_st1],1)
        pred_act = torch.softmax(self.inverse_model(phi_cat),1)

        pred_phi = self.forward_model(torch.cat([phi_st,act],1))
        return pred_act,pred_phi,phi_st

    def get_intrinsic_reward(self,obs,act,next_obs):
        pred_act,pred_phi,phi_st = self.forward(obs,act,next_obs)
        ri = (pred_phi-phi_st).pow(2).sum()
        return ri.detach().item()



