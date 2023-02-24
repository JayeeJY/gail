import gym
import torch
import random, math
from torch import nn
import numpy as np
from torch.distributions import Normal
import torch.nn.functional as F
from bc.dataset.collector import Collector
from bc.utils import misc
from copy import deepcopy
from bc.dataset import ImitationDataset
from configs.bc import train_ingredient
from sacred import Experiment
from bc.model import utils, log
import sys
import os
import pickle
from numbers import Real

from rewardoptim.Optimizer import Optimizer
if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

import json

LR = 1e-3
MAX_EPISODES = 10000
MAX_EPISODES_LEN = 200
INIT_STD = 0.075
GAMMA = 0.99
UPDATE_INTERVAL = 2100
LOG_INTERVAL = 20
CLIP = 0.2
EPOCH = 32
BATCH_SIZE = 200
N_ACTIONS = 3
N_STATE_DIM = 36
MAX_ACTION = 3

ex = Experiment('train', ingredients=[train_ingredient])
log_path = "/mnt/c/SMART/transformer/transformercombineskills/rlbc/rewardtrain/rewardvisual"
os.environ['RLBC_ROOT'] = "/mnt/c/SMART/transformer/transformercombineskills/rlbc"
os.environ['RLBC_MODELS'] = "/mnt/c/SMART/transformer/transformercombineskills/rlbc/models"
os.environ['RLBC_DATA'] = "/mnt/c/SMART/transformer/transformercombineskills/rlbc/datanew"
sys.path.append("/mnt/c/SMART/transformer/transformercombineskills/rlbc")
img_path="/mnt/c/SMART/transformer/transformercombineskills/rlbc/rewardtrain/images"
data_path = "/mnt/c/SMART/transformer/transformercombineskills/rlbc/rewardtrain/datarcnn"


class GripperActorNN(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.hidden1 = nn.Linear(state_dim, 512)
        self.hidden2 = nn.Linear(512, 256)
        self.hidden3 = nn.Linear(256, 128)
        self.hidden4 = nn.Linear(128, 64)
        self.classify_layer = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden2(x))
        x = torch.tanh(self.hidden3(x))
        x = torch.tanh(self.hidden4(x))
        prob = self.classify_layer(x)
        return prob

class ActorNN(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(ActorNN, self).__init__()
        self.max_action = max_action
        self.hidden1 = nn.Linear(state_dim, 512)
        self.hidden2 = nn.Linear(512, 256)
        self.hidden3 = nn.Linear(256, 128)
        self.hidden4 = nn.Linear(128, 64)
        self.mu_layer = nn.Linear(64, action_dim)


    def forward(self, x):
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden2(x))
        x = torch.tanh(self.hidden3(x))
        x = torch.tanh(self.hidden4(x))
        mu = torch.tanh(self.mu_layer(x))*0.075
        return mu

class stdNN(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.hidden1 = nn.Linear(state_dim, 512)
        self.hidden2 = nn.Linear(512, 256)
        self.hidden3 = nn.Linear(256, 128)
        self.hidden4 = nn.Linear(128, 64)
        self.std_layer = nn.Linear(64, action_dim)


    def forward(self, x):
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden2(x))
        x = torch.tanh(self.hidden3(x))
        x = torch.tanh(self.hidden4(x))
        std = torch.sigmoid(self.std_layer(x))*0.075
        return std


class CriticNN(nn.Module):

    def __init__(self, state_dim):
        super(CriticNN, self).__init__()
        self.hidden1 = nn.Linear(state_dim, 512)
        self.hidden2 = nn.Linear(512, 256)
        self.hidden3 = nn.Linear(256, 128)
        self.hidden4 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, state):
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        value = self.output(x)
        return value


class DiscriminatorNN(nn.Module):

    def __init__(self, sa_dim):
        super(DiscriminatorNN, self).__init__()
        self.hidden1 = nn.Linear(sa_dim, 512)
        self.hidden2 = nn.Linear(512, 256)
        self.hidden3 = nn.Linear(256, 128)
        self.hidden4 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden2(x))
        x = torch.tanh(self.hidden3(x))
        x = torch.tanh(self.hidden4(x))
        return self.output(x)


class PPO_Agent():

    def __init__(self, state_dim, action_dim, max_action, init_std, lr, gamma, epoch, clip):

        self.policy_actor = ActorNN(state_dim, action_dim, max_action)
        self.std_actor = stdNN(state_dim, action_dim)
        self.gripper = GripperActorNN(state_dim)
        self.policy_critic = CriticNN(state_dim)

        self.base_actor = ActorNN(state_dim, action_dim, max_action)
        self.base_std = stdNN(state_dim, action_dim)
        self.base_actor.load_state_dict(self.policy_actor.state_dict())
        self.base_actor.eval()
        self.base_std.load_state_dict(self.std_actor.state_dict())
        self.std_actor.eval()

        self.base_critic = CriticNN(state_dim)
        self.base_critic.load_state_dict(self.policy_critic.state_dict())
        self.base_critic.eval()

        self.optimizer = torch.optim.Adam([
            {'params': self.policy_actor.parameters(), 'lr': lr},
            {'params': self.policy_critic.parameters(), 'lr': 2 * lr},
            {'params': self.std_actor.parameters(), 'lr': lr}
        ])
        self.gripper_optimizer = torch.optim.Adam([
            {'params': self.gripper.parameters(), 'lr': lr}
        ])

        self.action_std = init_std
        self.init_std = init_std
        self.loss_fn = F.mse_loss
        self.gamma = gamma
        self.epoch = epoch
        self.clip = clip
        self.gail_epoch = epoch

    def select_action(self, state):
        #mu = self.base_actor(torch.FloatTensor(state))
        mu = self.base_actor(state)
        #std = self.action_std * torch.ones_like(mu)
        std = self.base_std(state)
        #dist = Normal(mu, std)
        #action = dist.sample()
        dist = Normal(mu, std)
        action = dist.rsample()
        logprob = dist.log_prob(action)
        return action.detach().cpu().numpy(), logprob.detach().cpu().numpy()

    def generate_gripper(self, state):
        prob = self.gripper(state)
        if prob[0] >= prob[1]:
                grip = 0
        else:
                grip = -2
        return grip,prob

    def decay_action_std(self, eps, min_std):
        self.action_std = self.init_std - eps / 3000.0
        if (self.action_std <= min_std):
            self.action_std = min_std

    def compute_returns(self, reward, done):
        returns = []
        R = 0.0
        for r, d in zip(reward[::-1], done[::-1]):
            if d is True:
                R = 0.0
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float)
        return (returns - returns.mean()) / (returns.std() + 1e-8)

    def update(self, unzipped_rollout):
        # states, actions, logprob_base, rewards, dones = map(np.stack, zip(*rollout))
        states, actions, logprob_base, rewards, dones = unzipped_rollout

        states = torch.tensor(states, dtype=torch.float).detach()
        actions = torch.tensor(actions, dtype=torch.float).detach()
        logprob_base = torch.FloatTensor(logprob_base.sum(-1)).detach().cuda()

        returns = self.compute_returns(rewards, dones)

        for _ in range(self.epoch):
            mu = self.policy_actor(states)
            #std = self.action_std * torch.ones_like(mu)
            std = self.std_actor(states)
            #dist = Normal(mu, std)
            var = (std ** 2)
            log_scale = math.log(std) if isinstance(std, Real) else std.log()
            logprob_policy = -((actions - mu) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
            logprob_policy = logprob_policy.cuda()
            #logprob_policy = dist.log_prob(actions).cuda()

            sv_policy = self.policy_critic(states).squeeze()

            advantages = returns - sv_policy.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ratio = torch.exp(logprob_policy.sum(-1) - logprob_base)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * F.mse_loss(sv_policy, returns)
            loss = loss.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.base_actor.load_state_dict(self.policy_actor.state_dict())
        self.base_std.load_state_dict(self.std_actor.state_dict())

    def gripperupdate(self, grip_v, exp_actions):

            exppar_actions = torch.FloatTensor(exp_actions)
            grippar_v = torch.cat(grip_v, dim=0)
            clsloss = F.binary_cross_entropy_with_logits(grippar_v,exppar_actions.cuda())
            self.gripper_optimizer.zero_grad()
            clsloss.backward()
            self.gripper_optimizer.step()

            return clsloss








class Discriminator():

    def __init__(self, state_dim, action_dim, lr, epoch):
        self.disc = DiscriminatorNN(state_dim + action_dim)
        self.disc_optim = torch.optim.Adam(self.disc.parameters(), lr=lr)
        self.gail_epoch = epoch

    def update(self, expert_rollout, policy_rollout):
        loss = 0.0

        for _ in range(self.gail_epoch):
            expert_batch = random.sample(expert_rollout, BATCH_SIZE)
            policy_batch = random.sample(policy_rollout, BATCH_SIZE)

            expert_states, expert_actions, _, _, _ = map(torch.FloatTensor, map(np.stack, zip(*expert_batch)))
            policy_states, policy_actions, _, _, _ = map(torch.FloatTensor, map(np.stack, zip(*policy_batch)))

            expert_d = self.disc(torch.cat([expert_states.cuda(), expert_actions.cuda()], dim=1))
            policy_d = self.disc(torch.cat([policy_states.cuda(), policy_actions.cuda()], dim=1))

            #expert_loss = F.binary_cross_entropy_with_logits(expert_d, torch.ones(expert_d.size()))
            expert_loss = F.binary_cross_entropy_with_logits(expert_d, torch.zeros(expert_d.size()))
            #policy_loss = F.binary_cross_entropy_with_logits(policy_d, torch.zeros(policy_d.size()))
            policy_loss = F.binary_cross_entropy_with_logits(policy_d, torch.ones(policy_d.size()))

            gail_loss = expert_loss + policy_loss

            self.disc_optim.zero_grad()
            gail_loss.backward()
            self.disc_optim.step()

            loss += gail_loss.item()

        return loss / self.gail_epoch

    def predict_rewards(self, rollout):
        states, actions, logprob_base, rewards, dones = map(np.stack, zip(*rollout))
        with torch.no_grad():
            policy_mix = torch.cat([torch.FloatTensor(states), torch.FloatTensor(actions)], dim=1).cuda()
            policy_d = self.disc(policy_mix).squeeze()
            score = torch.sigmoid(policy_d)
            # gail_rewards = - (1-score).log()
            #gail_rewards = score.log() - (1 - score).log()
            gail_rewards = (-1) * torch.log(score)
            return (states, actions, logprob_base, gail_rewards.cpu().numpy(), dones)


# ------------------------------------------------------------------------------

@ex.automain
def main(model, dataset):
    torch.manual_seed(0)
    np.random.seed(0)
    #ckpt_path = "gailmain/ckpts"
    ckpt_path = "ckpts"
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)
    collect={}
    collect["folder"] = "pick_evaluationopt"
    collect["env"] = "UR5-Pick5RandCamEnv-v0"
    collect["db_type"] = "evaluation"
    collect["agent"] = "bc"
    collect["seed"] = 5001
    collect["episodes"] = 1
    collect["first_epoch"] = 64
    collect["iter_epoch"] = 4
    collect["workers"] = 3
    collect["max_steps"] = -1
    collect["timescale"]=60
    collect["skill_sequence"]=[]
    cpmodel = deepcopy(model)
    model,collect = misc.update_arguments(model=model, collect=collect)
    collector = Collector(**model,**collect, report_path=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpmodel["name"] = "pick_policy"
    cpmodel["action_space"] = "tool_lin"
    dataset["num_cameras"] = 5
    dataset["name"] = "pick_demos"
    cpmodel, dataset = misc.update_arguments(model=cpmodel, dataset=dataset)
    dataset_extra_args = dict(
        num_frames=cpmodel['num_frames'],
        channels=cpmodel['input_type'],
        action_space=cpmodel['action_space'],
        steps_action=cpmodel['steps_action'],
        num_signals=cpmodel['num_signals'],
        num_skills=cpmodel['num_skills'])
    im_dataset = ImitationDataset(**dataset, **dataset_extra_args)
    encoder, optimizer = utils.load_model("/mnt/c/SMART/transformer/transformercombineskills/rlbc/models/pick_policy", epoch=100, device=device)
    ckpt_path = os.path.join(ckpt_path, collect["env"])
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)
    encoder.set_eval()
    optimizer = Optimizer()
    optimizer.loadattributes(model)
    env = collector.env
    keys, scalars = im_dataset.get_keys_scalars()
    orgaction = scalars[keys[0]]['action']
    rlaction=deepcopy(orgaction)
    agent = PPO_Agent(state_dim=N_STATE_DIM, action_dim=N_ACTIONS, max_action=MAX_ACTION, init_std=INIT_STD, lr=LR,
                      gamma=GAMMA, epoch=EPOCH, clip=CLIP)
    disc = Discriminator(state_dim=N_STATE_DIM, action_dim=N_ACTIONS, lr=LR, epoch=EPOCH)
    # for name, param in agent.policy_net.actor.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
    #     print (param.dtype)



    len_episodes = []
    reward_per_eps = []
    gail_loss = []
    expert_rollout = []
    rollout = []
    curr_state = env.reset()
    t_per_eps = 0
    eps = 0
    reward_tmp = 0.0
    #grip_v = []
    rewardlog=[]
    mselosslog=[]
    while eps <= MAX_EPISODES:
        frame = optimizer.obtoframe(curr_state)
        frame = frame.unsqueeze(0)
        feature = encoder(frame).detach()
        feature = feature.squeeze()
        config = np.hstack((curr_state['joint_position'], curr_state['tool_position'], curr_state['tool_orientation'],
                            curr_state['linear_velocity']))
        config = torch.from_numpy(config).cuda()
        feature = torch.cat((feature, config), dim=0).float()
        action, logprob = agent.select_action(feature)
        #grip, prob = agent.generate_gripper(feature)
        gripaction=encoder.get_dict_action(frame)
        #rlaction['grip_velocity'] = grip
        rlaction['grip_velocity'] = gripaction['grip_velocity']
        rlaction['linear_velocity'] = action
        #action = np.insert(action,0,[gripaction[0]])
        #rlaction['grip_velocity'] = action[0]
        #rlaction['linear_velocity'] = action[1:4]
        next_state, reward, done, _ = env.step(rlaction)
        feature = feature.cpu().numpy()
        rollout.append((feature, action, logprob, reward, done))

        reward_tmp += reward
        t_per_eps += 1

        curr_state = next_state

        if done or t_per_eps > MAX_EPISODES_LEN:
            reward_per_eps.append(reward_tmp)
            reward_tmp = 0.0
            len_episodes.append(t_per_eps)
            t_per_eps = 0

            eps += 1
            curr_state = env.reset()

            #agent.decay_action_std(eps, 0.1)


            if eps % LOG_INTERVAL == 0:
                print('Episodes: ' + str(eps) + '  --------------------')
                print(np.mean(reward_per_eps[-100:]))
                rewardlog.append(np.mean(reward_per_eps[-100:]))

        if len(rollout) == UPDATE_INTERVAL:
            exp_actions = []
            pol_actions = []
            numaction=deepcopy(action)
            expert_idx = random.sample(range(22281),UPDATE_INTERVAL)
            for idx in expert_idx:
                file_name = 'demoimg_{}.pth'.format(idx)
                file_path = os.path.join(img_path, file_name)
                expframe = torch.load(file_path)
                expfeature = encoder(expframe).detach()
                expfeature = expfeature.squeeze()
                state = scalars[keys[idx]]['state']
                expconfig = np.hstack(
                    (state['joint_position'], state['tool_position'], state['tool_orientation'],
                     state['linear_velocity']))
                expconfig = torch.from_numpy(expconfig).cuda()
                expfeature = torch.cat((expfeature, expconfig), dim=0).float()
                polaction,pollogprob = agent.select_action(expfeature)
                #exploregrip, prob = agent.generate_gripper(expfeature)
                #grip_v.append(prob.unsqueeze(0))
                expfeature = expfeature.cpu().numpy()
                exaction = scalars[keys[idx]]['action']
                #numaction[0] = exaction['grip_velocity'][0]
                #numaction[1:4] = exaction['linear_velocity'][0:3]
                numaction = exaction['linear_velocity']
                expert_rollout.append((expfeature,numaction,0,0,False))
                exp_actions.append(numaction)
                pol_actions.append(polaction)



            #griploss = agent.gripperupdate(grip_v, exp_actions)
            lossfun = nn.MSELoss(reduction='sum')
            lossfun.to(device)
            exp_actions = torch.FloatTensor(np.array(exp_actions)).cuda()
            pol_actions = torch.FloatTensor(np.array(pol_actions)).cuda()
            mseloss = lossfun(pol_actions, exp_actions)
            mselosslog.append(mseloss.cpu().numpy())
            loss = disc.update(expert_rollout, rollout)
            gail_loss.append(loss)
            unzipped_rollout = disc.predict_rewards(rollout)
            agent.update(unzipped_rollout)
            del rollout[:]
            del expert_rollout[:]
            #del grip_v[:]
            with open(os.path.join(ckpt_path, "rewardlog.pkl"), "wb") as f:
                pickle.dump(rewardlog, f)
            with open(os.path.join(ckpt_path, "mselosslog.pkl"), "wb") as f:
                pickle.dump(mselosslog, f)
            torch.save(
                agent.policy_actor.state_dict(), os.path.join(ckpt_path, 'policy.pth')
            )
            torch.save(
                agent.policy_critic.state_dict(), os.path.join(ckpt_path, 'value.pth')
            )
            torch.save(
                disc.disc.state_dict(), os.path.join(ckpt_path, 'discriminator.pth')
            )
            torch.save(
                agent.std_actor.state_dict(), os.path.join(ckpt_path, 'std.pth')
            )

            #torch.save(
                #agent.gripper.state_dict(), os.path.join(ckpt_path, 'gripper.pth')
            #)
            print(
                "Episodes: {},   Gail Loss: {}, MSE loss:{}"
                .format(eps, loss, mseloss)
            )

    env.close()
