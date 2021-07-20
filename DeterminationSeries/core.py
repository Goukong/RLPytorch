import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np

def combine_shape(length,shape = None):
    if shape is None:
        return (length,)
    return (length,shape) if np.isscalar(shape) else (length,*shape)

def mlp(sizes,activation,output_activation = nn.Identity):
    layers = []
    for i in range(len(sizes)-1):
        act = activation if i < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[i],sizes[i+1]),act()]
    return nn.Sequential(*layers)

class Actor(nn.Module):
    def __init__(self,obs_dim,act_dim,hidden_layers,activation,act_limit):
        super().__init__()
        self.pi = mlp([obs_dim]+list(hidden_layers)+[act_dim],activation=activation,output_activation=nn.Tanh)
        self.act_limit = act_limit

    def forward(self,obs):
        return self.act_limit * self.pi(obs)

#与V不同，输入是s和a
#SAC和DDPG用的Q网络都是相同的
class QCritic(nn.Module):
    def __init__(self,obs_dim,act_dim,hidden_layers,activation):
        super().__init__()
        self.q = mlp([obs_dim+act_dim] + list(hidden_layers) + [1],activation=activation)
    def forward(self,obs,act):
        q = self.q(torch.cat([obs,act],dim=-1))
        return torch.squeeze(q,-1)

class ActorCritic(nn.Module):
    def __init__(self,obs_space,act_space,hidden_layers=(256,256),activation=nn.ReLU):
        super(ActorCritic, self).__init__()
        obs_dim = obs_space.shape[0]
        act_dim = act_space.shape[0]
        act_limit = act_space.high[0]

        self.pi = Actor(obs_dim,act_dim,hidden_layers,activation,act_limit)
        self.q = QCritic(obs_dim,act_dim,hidden_layers,activation)
    def act(self,obs):
        with torch.no_grad():
            return self.pi(obs).numpy()

class SquashedGaussianActor(nn.Module):
    def __init__(self,obs_dim,act_dim,hidden_layers,activation,act_limit):
        super(SquashedGaussianActor,self).__init__()
        #输入状态，输出特征；类似dueling结构，为了后续计算mu和std
        self.net = mlp([obs_dim]+list(hidden_layers),activation,activation)
        #无激活函数
        self.mu_layer = nn.Linear(hidden_layers[-1],act_dim)
        self.log_std_layer = nn.Linear(hidden_layers[-1],act_dim)
        self.act_limit = act_limit
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self,obs,deterministic=False,with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clip(log_std,self.log_std_min,self.log_std_max)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu,std)

        #测试时的动作是确定的，不需要添加高斯噪声
        if deterministic:
            pi_action = mu
        else:
            #rsample 的含义是为了让Q值网络更新时也可以将梯度通过动作传递过来
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)

            #因为后面会用tanh，这时的随机变量就换元了，所以要进行处理
            #test = torch.tanh(pi_action)

            #这样写会梯度消失
            #logp_pi -=  (torch.log(1-torch.tanh(pi_action))).sum(axis=1)
            logp_pi -= (2*(np.log(2)-pi_action-F.softplus(-2*pi_action))).sum(axis=1)
        else:
            #同样为了测试
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        return pi_action,logp_pi

class SACActorCritic(nn.Module):
    def __init__(self,obs_space,act_space,hidden_layers=(256,256),activation=nn.ReLU):
        super(SACActorCritic, self).__init__()
        obs_dim = obs_space.shape[0]
        act_dim = act_space.shape[0]

        act_limit = act_space.high[0]

        self.pi = SquashedGaussianActor(obs_dim,act_dim,hidden_layers,activation,act_limit)
        self.q1 = QCritic(obs_dim,act_dim,hidden_layers,activation)
        self.q2 = QCritic(obs_dim,act_dim,hidden_layers,activation)

    def act(self,obs,deterministic=False):
        with torch.no_grad():
            a,_ = self.pi(obs,deterministic,False)
            return a.numpy()



