import torch
import gym
import numpy as np
from Buffer import Buffer
import core
from tensorboardX import SummaryWriter
import os

def ppo(env_fn,actor_critic = core.MLPActorCritic,ac_kwards = dict(),seed = 0,
        step_per_epoch = 5000,epochs = 50,gamma = 0.96,lam = 0.98,
        pi_lr = 3e-4,vf_lr = 1e-3,train_pi_iter=8,train_v_iter = 8,clip_ratio=0.2,
        max_ep_len = 1000,save_freq=10,intrinsic_weight=0.01):
    torch.manual_seed(seed)
    np.random.seed(seed)

    path = 'run/PPO/'
    if not os.path.exists(path):
        os.makedirs(path)
    logger = SummaryWriter(path, comment='PPO')

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    ac = actor_critic(env.observation_space, env.action_space, **ac_kwards)

    buf = Buffer(step_per_epoch, obs_dim, act_dim, gamma, lam)

    pi_optimizer = torch.optim.Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = torch.optim.Adam(ac.v.parameters(), lr=vf_lr)

    def compute_loss_pi(data):
        obs,act,adv,logp_old = data['obs'],data['act'],data['adv'],data['logp']
        pi,logp = ac.pi(obs,act)
        ratio = torch.exp(logp-logp_old)
        clip_adv = torch.clip(ratio,1-clip_ratio,1+clip_ratio) * adv

        ent = pi.entropy()
        kl = logp_old - logp
        loss_pi = -(torch.min(ratio*adv,clip_adv)).mean()

        return loss_pi,ent.mean().item(),kl

    def compute_loss_v(data):
        obs,ret = data['obs'],data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    def update():
        data = buf.get()

        pi_l_old,ent,kl = compute_loss_pi(data)
        v_l_old = compute_loss_v(data)

        for i in range(train_pi_iter):
            pi_optimizer.zero_grad()
            loss_pi,ent,kl = compute_loss_pi(data)
            # if kl > 1.5 * 0.01:
            #     break
            loss_pi.backward()
            pi_optimizer.step()

        for i in range(train_v_iter):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            vf_optimizer.step()


        return loss_pi,loss_v,ent,kl.mean().item()

    o, ep_rew, ep_len = env.reset(), 0, 0


    for epoch in range(epochs):
        ret_list = []
        #一个epoch是一个策略
        #一次range的迭代收集的是同一个策略收集的
        for i in range(step_per_epoch):
            #env.render()
            a,v,logp = ac.step(torch.as_tensor(o,dtype=torch.float32))
            next_o,r,d,_ = env.step(a)

            ep_rew += r
            ep_len += 1

            buf.store(o,a,r,v,logp,next_o)

            o = next_o

            #这一栏代表能容忍场景最大的重复次数
            timeout = ep_len == max_ep_len
            #场景结束，或是超时都会停止
            terminal = d or timeout
            #buffer满了也要终止
            epoch_ended = i == step_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    #ep_len代表该场景下的执行次数
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                if timeout or epoch_ended:
                    _,v,_ = ac.step(torch.as_tensor(o,dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                ret_list.append(ep_rew)
                o,ep_rew,ep_len = env.reset(),0,0

        if (epoch % save_freq == 0 and epoch != 0) or (epoch == epochs - 1):
            torch.save(ac,'PPO_model.pth')
        # 记录还没写 logger
        loss_pi,loss_v,ent,kl = update()
        ret_mean = np.mean(ret_list)
        logger.add_scalar('loss_pi:',loss_pi,epoch)
        logger.add_scalar('loss_v:',loss_v,epoch)
        logger.add_scalar('ret_mean:',ret_mean,epoch)
        logger.add_scalar('ent:',ent,epoch)
        logger.add_scalar('kl:',kl,epoch)

        print('epoch: %3d \t ret_mean %.3f' %
              (epoch, ret_mean))
    env.close()

if __name__ == '__main__':
    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument('--env',type=str,default='CartPole-v0')
    parse.add_argument('--hid',type=int,default=32)
    parse.add_argument('--l',type=int,default=2)
    parse.add_argument('--gamma',type=float,default=0.98)
    parse.add_argument('--seed','-s',type=int,default=0)
    parse.add_argument('--bufSize',type=int,default=2048)
    parse.add_argument('--epochs',type=int,default=100)
    args = parse.parse_args()

    ppo(lambda : gym.make(args.env),actor_critic=core.MLPActorCritic,
        ac_kwards=dict(hidden_sizes=[args.hid]*args.l),step_per_epoch=args.bufSize,
        gamma=args.gamma,seed=args.seed,epochs=args.epochs)

