import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from eval_metrics import *
from model.DynamicGRU import DynamicGRU
from torch.distributions import Categorical
from collections import namedtuple

from utils import *

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class hpg(nn.Module):
    def __init__(self, state_dim, act_dim, num_items, config, device):  # 1100，4
        super(hpg, self).__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.gamma = config.gamma
        self.args = config
        self.device = device
        self.item_embeddings = nn.Embedding(num_items, config.d).to(device)
        self.embeds = load_embed(config.dataset)

        self.l1 = nn.Linear(state_dim, config.hidden[0])
        self.l2 = nn.Linear(config.hidden[0], config.hidden[1])
        self.actor = nn.Linear(config.hidden[1], act_dim)
        self.critic = nn.Linear(config.hidden[1], 1)

        self.saved_actions = []
        self.rewards = []
        self.entropy = []

        self.actions_step = 0
        self._batch_curr_state = None
        self._done = False

    def forward(self, inputs):
        state = inputs  # state: [bs, state_dim], act_mask: [bs, act_dim]
        x = self.l1(state)
        x = F.dropout(F.elu(x), p=0.5)
        out = self.l2(x)
        x = F.dropout(F.elu(out), p=0.5)

        actor_logits = self.actor(x)
        act_probs = F.softmax(actor_logits, dim=-1)  # Tensor of [bs, act_dim]

        state_values = self.critic(x)  # Tensor of [bs, 1]
        return act_probs, state_values

    def select_action(self, batch_state, batch_uids, kg_actions, seq_actions, device):
        state = []
        for i in range(len(batch_uids)):
            seq_actions_idx = torch.from_numpy(np.atleast_2d(seq_actions[batch_uids[i]])).type(torch.LongTensor).to(device)
            seq_actions_embeddings = self.item_embeddings(seq_actions_idx)

            one_state = np.concatenate([batch_state[i], self.embeds[PRODUCT][kg_actions[batch_uids[i]][self.actions_step]-1], seq_actions_embeddings[0][self.actions_step].cpu().data.numpy().tolist()])
            state.append(one_state)
        state = np.array(state)
        state = torch.FloatTensor(state).to(device)  # Tensor [bs, state_dim]
        probs, value = self(state)  # act_probs: [bs, act_dim], state_value: [bs, 1]
        m = Categorical(probs)
        acts = m.sample()  # Tensor of [bs, ], requires_grad=False

        self.saved_actions.append(SavedAction(m.log_prob(acts), value))
        self.entropy.append(m.entropy())
        self.actions_step += 1
        if self.actions_step == 5:
            self._done = True

        return acts.cpu().numpy().tolist()

    def select_actions(self, batch_state, batch_uids, kg_actions, seq_actions, device):
        state = []
        batch_states = []
        for i in range(len(batch_uids)):
            seq_actions_idx = torch.from_numpy(np.atleast_2d(seq_actions)).type(torch.LongTensor).to(device)
            seq_actions_embeddings = self.item_embeddings(seq_actions_idx)

            one_state = np.concatenate([batch_state[i], self.embeds[PRODUCT][kg_actions[i]-1], seq_actions_embeddings[0][i].cpu().data.numpy().tolist()])
            batch_states.append(one_state.tolist())
            state.append(one_state)
        state = np.array(state)
        state = np.pad(state, ((0, 0), (0, self.state_dim - len(state[0]))), 'constant')
        state = torch.FloatTensor(state).to(device)  # Tensor [bs, state_dim]

        probs, value = self(state)  # act_probs: [bs, act_dim], state_value: [bs, 1]
        m = Categorical(probs)
        acts = m.sample()  # Tensor of [bs, ], requires_grad=False

        self.saved_actions.append(SavedAction(m.log_prob(acts), value))
        self.entropy.append(m.entropy())
        self.actions_step += 1
        if self.actions_step == 5:
            self._done = True
        return batch_states, acts.cpu().numpy().tolist(), probs

    def select_actions_short(self, batch_state, batch_uids, kg_actions, seq_actions, device):
        state = []
        for i in range(len(batch_uids)):
            seq_actions_idx = torch.from_numpy(np.atleast_2d(seq_actions)).type(torch.LongTensor).to(device)
            seq_actions_embeddings = self.item_embeddings(seq_actions_idx)

            one_state = np.concatenate([batch_state[i], self.embeds[PRODUCT][kg_actions[i]-1], seq_actions_embeddings[0][i].cpu().data.numpy().tolist()])
            state.append(one_state)
        state = np.array(state)
        state = torch.FloatTensor(state).to(device)  # Tensor [bs, state_dim]

        probs, value = self(state)  # act_probs: [bs, act_dim], state_value: [bs, 1]
        m = Categorical(probs)
        acts = m.sample()  # Tensor of [bs, ], requires_grad=False

        self.saved_actions.append(SavedAction(m.log_prob(acts), value))
        self.entropy.append(m.entropy())
        self.actions_step += 1
        if self.actions_step == 5:
            self._done = True
        return acts.cpu().numpy().tolist(), probs

    def generate_action(self, batch_state, batch_uids, kg_actions, seq_actions, device):
        state = []
        acts = []
        batch_states = []
        for i in range(len(batch_uids)):
            seq_actions_idx = torch.from_numpy(np.atleast_2d(seq_actions[batch_uids[i]])).type(torch.LongTensor).to(device)
            seq_actions_embeddings = self.item_embeddings(seq_actions_idx)

            one_state = np.concatenate([batch_state[i], self.embeds[PRODUCT][kg_actions[batch_uids[i]][self.actions_step]-1], seq_actions_embeddings[0][self.actions_step].cpu().data.numpy().tolist()])
            batch_states.append(one_state.tolist())
            state.append(one_state)
        state = np.array(state)
        state = np.pad(state, ((0, 0), (0, self.state_dim - len(state[0]))), 'constant')
        state = torch.FloatTensor(state).to(device)  # Tensor [bs, state_dim]
        probs, value = self(state)  # act_probs: [bs, act_dim], state_value: [bs, 1]
        # print(probs)
        topk_probs, topk_idxs = torch.topk(probs, 2, dim=1)
        for i in range(len(batch_uids)):
            acts.append(topk_idxs[i][0].cpu().numpy().tolist())
        # m = Categorical(probs)
        # acts = m.sample()  # Tensor of [bs, ], requires_grad=False

        self.actions_step += 1
        if self.actions_step == 10:
            self._done = True

        return acts, batch_states

    def generate_action_short(self, batch_state, batch_uids, kg_actions, seq_actions, device):
        state = []
        acts = []
        for i in range(len(batch_uids)):
            seq_actions_idx = torch.from_numpy(np.atleast_2d(seq_actions[batch_uids[i]])).type(torch.LongTensor).to(device)
            seq_actions_embeddings = self.item_embeddings(seq_actions_idx)

            one_state = np.concatenate([batch_state[i], self.embeds[PRODUCT][kg_actions[batch_uids[i]][self.actions_step]-1], seq_actions_embeddings[0][self.actions_step].cpu().data.numpy().tolist()])
            state.append(one_state)
        state = np.array(state)
        state = torch.FloatTensor(state).to(device)  # Tensor [bs, state_dim]
        probs, value = self(state)  # act_probs: [bs, act_dim], state_value: [bs, 1]
        # print(probs)
        topk_probs, topk_idxs = torch.topk(probs, 2, dim=1)
        for i in range(len(batch_uids)):
            acts.append(topk_idxs[i][0].cpu().numpy().tolist())
        # m = Categorical(probs)
        # acts = m.sample()  # Tensor of [bs, ], requires_grad=False

        self.actions_step += 1
        if self.actions_step == 10:
            self._done = True

        return acts

    def update(self, optimizer, device, ent_weight):
        if len(self.rewards) <= 0:
            del self.rewards[:]
            del self.saved_actions[:]
            del self.entropy[:]
            return 0.0, 0.0, 0.0

        batch_rewards = np.vstack(self.rewards).T  # numpy array of [bs, #steps]
        batch_rewards = torch.FloatTensor(batch_rewards).to(device)
        num_steps = batch_rewards.shape[1]
        # for i in range(1, num_steps):
        #     batch_rewards[:, num_steps - i - 1] += self.gamma * batch_rewards[:, num_steps - i]

        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        for i in range(0, num_steps):
            log_prob, value = self.saved_actions[i]  # log_prob: Tensor of [bs, ], value: Tensor of [bs, 1]
            advantage = batch_rewards[:, i] - value.squeeze(1)  # Tensor of [bs, ]
            actor_loss += -log_prob * advantage.detach()  # Tensor of [bs, ]
            critic_loss += advantage.pow(2)  # Tensor of [bs, ]
            entropy_loss += -self.entropy[i]  # Tensor of [bs, ]
        actor_loss = actor_loss.mean()
        critic_loss = critic_loss.mean()
        entropy_loss = entropy_loss.mean()
        loss = actor_loss + critic_loss + ent_weight * entropy_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del self.rewards[:]
        del self.saved_actions[:]
        del self.entropy[:]

        return loss.item(), actor_loss.item(), critic_loss.item(), entropy_loss.item()


    def reset(self):
        self.actions_step = 0
        self._done = False


        return