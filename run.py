import argparse
import logging
from time import time

import torch

from train_agent import ActorCritic

import json

import random

from data import Amazon
from HPG import *
from model.KERL import *
from hpg_env import *
from kg_env import BatchKGEnvironment
from utils import *
from enhancement import *

import datetime
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

device = torch.device('cuda:0')

class ACDataLoader(object):
    def __init__(self, uids, batch_size):
        self.uids = np.array(uids)
        self.num_users = len(uids)
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self._rand_perm = np.random.permutation(self.num_users)
        self._start_idx = 0
        self._has_next = True

    def has_next(self):
        return self._has_next

    def get_batch(self):
        if not self._has_next:
            return None
        # Multiple users per batch
        end_idx = min(self._start_idx + self.batch_size, self.num_users)
        batch_idx = self._rand_perm[self._start_idx:end_idx]
        batch_uids = self.uids[batch_idx]
        self._has_next = self._has_next and end_idx < self.num_users
        self._start_idx = end_idx
        return batch_uids.tolist()

def eh_beam_search(env, model, uids, device, topk, eh_product):
    def _batch_acts_to_masks(batch_acts):
        batch_masks = []
        for acts in batch_acts:
            num_acts = len(acts)
            act_mask = np.zeros(model.act_dim, dtype=np.uint8)
            act_mask[:num_acts] = 1
            batch_masks.append(act_mask)
        return np.vstack(batch_masks)

    state_pool = env.reset(uids)  # numpy of [bs, dim]
    path_pool = env._batch_path  # list of list, size=bs
    new_path = path_pool[0] + [(PURCHASE, PRODUCT, eh_product-1)]
    path_pool = [new_path]
    state_pool = env._batch_get_state(path_pool)
    probs_pool = [[] for _ in uids]
    model.eval()
    for hop in range(3):
        state_tensor = torch.FloatTensor(state_pool).to(device)
        acts_pool = env._batch_get_actions(path_pool, False)  # list of list, size=bs
        actmask_pool = _batch_acts_to_masks(acts_pool)  # numpy of [bs, dim]
        actmask_tensor = torch.ByteTensor(actmask_pool).to(device)
        probs, _ = model((state_tensor, actmask_tensor))  # Tensor of [bs, act_dim]
        probs = probs + actmask_tensor.float()  # In order to differ from masked actions
        topk_probs, topk_idxs = torch.topk(probs, topk[hop], dim=1)  # LongTensor of [bs, k]
        topk_idxs = topk_idxs.detach().cpu().numpy()
        topk_probs = topk_probs.detach().cpu().numpy()

        new_path_pool, new_probs_pool = [], []
        for row in range(topk_idxs.shape[0]):
            path = path_pool[row]
            probs = probs_pool[row]
            for idx, p in zip(topk_idxs[row], topk_probs[row]):
                if idx >= len(acts_pool[row]):  # act idx is invalid
                    continue
                relation, next_node_id = acts_pool[row][idx]  # (relation, next_node_id)
                if relation == SELF_LOOP:
                    next_node_type = path[-1][1]
                else:
                    next_node_type = KG_RELATION[path[-1][1]][relation]
                new_path = path + [(relation, next_node_type, next_node_id)]
                new_path_pool.append(new_path)
                new_probs_pool.append(probs + [p])
        path_pool = new_path_pool
        probs_pool = new_probs_pool
        if hop < 2:
            state_pool = env._batch_get_state(path_pool)

    return path_pool, probs_pool

def evaluate(topk_matches, test_user_products):
    """Compute metrics for predicted recommendations.
    Args:
        topk_matches: a list or dict of product ids in ascending order.
    """
    invalid_users = []
    # Compute metrics
    precisions, recalls, ndcgs, hits = [], [], [], []
    test_user_idxs = len(test_user_products)
    for uid in range(test_user_idxs):
        if uid not in topk_matches or len(topk_matches[uid]) < 5:
            invalid_users.append(uid)
            continue
        pred_list, rel_set = topk_matches[uid][::-1], test_user_products[uid]
        if len(pred_list) == 0:
            continue

        dcg = 0.0
        hit_num = 0.0
        for i in range(len(pred_list)):
            if pred_list[i] in rel_set:
                dcg += 1. / (log(i + 2) / log(2))
                hit_num += 1
        # idcg
        idcg = 0.0
        for i in range(min(len(rel_set), len(pred_list))):
            idcg += 1. / (log(i + 2) / log(2))
        ndcg = dcg / idcg
        recall = hit_num / len(rel_set)
        precision = hit_num / len(pred_list)
        hit = 1.0 if hit_num > 0.0 else 0.0

        ndcgs.append(ndcg)
        recalls.append(recall)
        precisions.append(precision)
        hits.append(hit)

    avg_precision = np.mean(precisions) * 100
    avg_recall = np.mean(recalls) * 100
    avg_ndcg = np.mean(ndcgs) * 100
    avg_hit = np.mean(hits) * 100
    print('NDCG={:.3f} |  Recall={:.3f} | HR={:.3f} | Precision={:.3f} | Invalid users={}'.format(
            avg_ndcg, avg_recall, avg_hit, avg_precision, len(invalid_users)))


def eval_hpg(train_data, test_data, test_set, num_users, num_items, config):
    #load models
    args_PGPR = json.load(open('./config/pgpr.json', 'rt'))
    env = BatchKGEnvironment(args_PGPR['dataset'], args_PGPR['max_acts'], max_path_len=args_PGPR['max_path_len'],
                             state_history=args_PGPR['state_history'])
    pretrain_sd = torch.load(args_PGPR['log_dir'] + '/policy_model_epoch_50.ckpt')
    model = ActorCritic(env.state_dim, env.act_dim, gamma=args_PGPR['gamma'], hidden_sizes=args_PGPR['hidden']).to(device)
    model_sd = model.state_dict()
    model_sd.update(pretrain_sd)
    model.load_state_dict(model_sd)

    args_KERL = json.load(open('./config/kerl.json', 'rt'))
    pretrain_sd = torch.load(args_KERL['log_dir'] + '/model_epoch_35.ckpt') #改这里
    seq_model = kerl(num_users, num_items, args_KERL, device).to(device)
    model_sd = seq_model.state_dict()
    model_sd.update(pretrain_sd)
    seq_model.load_state_dict(model_sd)

    uids = list(env.kg(USER).keys())
    dataloader = ACDataLoader(uids, config.batch_size)
    env_hpg = BatchHPGEnvironment(args_PGPR['dataset'], num_users, num_items, train_data, test_data, config.train_lenth)
    pretrain_sd = torch.load('./tmp/Amazon_Beauty/old_hpg_train' + '/policy_model_epoch_10.ckpt')
    hpg_model = hpg(config.d*3, 2, num_items, config, device).to(device)
    model_sd = hpg_model.state_dict()
    model_sd.update(pretrain_sd)
    hpg_model.load_state_dict(model_sd)
    users = 0
    pretrain_sd = torch.load('./tmp/Amazon_Beauty/old_hpg_train' + '/model_enh_epoch_10.ckpt')
    data_enh = enhancement(args_PGPR['dataset'], num_items, test_data, config, device)
    model_sd = data_enh.state_dict()
    model_sd.update(pretrain_sd)
    data_enh.load_state_dict(model_sd)

    hpg_model.eval()
    model.eval()
    seq_model.eval()
    data_enh.eval()

    pred_labels = {}
    kg_pred_labels = {}
    seq_pred_labels = {}
    eh_kg_pred_labels = {}
    eh_seq_pred_labels = {}
    for uid in range(num_users):
        pred_labels[uid] = []
        kg_pred_labels[uid] = []
        seq_pred_labels[uid] = []
        eh_kg_pred_labels[uid] = []
        eh_seq_pred_labels[uid] = []
    dataloader.reset()
    uuid = 0
    while dataloader.has_next():
        batch_uids = dataloader.get_batch()  # 随机取一批用户id
        #batch_uids = [0, 1, 10, 100, 1000, 10000, 10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008, 10009, 1001]
        batch_state = env_hpg.user_embeds[batch_uids] #获取用户向量
        hpg_model.reset()
        env_hpg.reset(batch_uids)
        kg_enh = [0 for uid in range(num_users)]
        seq_enh = [0 for uid in range(num_users)]
        kg_actions, seq_actions = env_hpg.batch_actions(env, model, batch_uids, device, args_PGPR['topk'],
                                                            seq_model, 'score')

        step = 0
        while step < 5:
            if step < 5:
                for uid in batch_uids:
                    if (kg_actions[uid][hpg_model.actions_step] in seq_actions[uid][:hpg_model.actions_step + 1]) and \
                            kg_enh[uid] == 0:
                        paths, probs = eh_beam_search(env, model, [uid], device, args_PGPR['topk'],
                                                      kg_actions[uid][hpg_model.actions_step])
                        kg_actions_eh = env_hpg.generate_kg_action_enh(paths, probs, [uid], 'score', kg_actions[uid][:hpg_model.actions_step+1])
                        for j in range(len(kg_actions_eh[uid])):
                            kg_actions_eh[uid][j] += 1
                        if len(kg_actions[uid][hpg_model.actions_step:]) != len(kg_actions_eh[uid][:config.train_lenth-hpg_model.actions_step]):
                            print(kg_actions[uid])
                            print(kg_actions_eh[uid])
                        else:
                            prediction_score, orgin, batch_targets, Reward = data_enh.enh_train(uid
                                                                                            , num_items
                                                                                            , kg_actions[uid][
                                                                                              hpg_model.actions_step:]
                                                                                            , kg_actions_eh[uid][
                                                                                              :config.train_lenth - hpg_model.actions_step]
                                                                                            ,
                                                                                            config.train_lenth - hpg_model.actions_step
                                                                                            , 'kg'
                                                                                            , test_data[uid]
                                                                                            , device)

                            if Reward.cpu().data.numpy()[0] * batch_targets.cpu().data.numpy()[0] == 1:
                                kg_enh[uid] = 1
                                kg_actions[uid][hpg_model.actions_step:] = kg_actions_eh[uid][
                                                                       :config.train_lenth - hpg_model.actions_step]


                    if (seq_actions[uid][hpg_model.actions_step] in kg_actions[uid][:hpg_model.actions_step + 1]) and \
                             seq_enh[uid] == 0:
                        seq_actions_eh = env_hpg.generate_seq_action_enh(seq_model, env_hpg.train_data,
                                                                         env_hpg.test_data,
                                                                         [uid], device,
                                                                         seq_actions[uid][hpg_model.actions_step], seq_actions[uid][:hpg_model.actions_step+1])

                        prediction_score, orgin, batch_targets, Reward = data_enh.enh_train(uid
                                                                                            , num_items
                                                                                            , seq_actions[uid][
                                                                                              hpg_model.actions_step:]
                                                                                            , seq_actions_eh[uid][
                                                                                              :config.train_lenth - hpg_model.actions_step]
                                                                                            ,
                                                                                            config.train_lenth - hpg_model.actions_step
                                                                                            , 'seq'
                                                                                            , test_data[uid]
                                                                                            , device)

                        if Reward.cpu().data.numpy()[0] * batch_targets.cpu().data.numpy()[0] == 1:
                            seq_enh[uid] = 1
                            seq_actions[uid][hpg_model.actions_step:] = seq_actions_eh[uid][
                                                                        :config.train_lenth - hpg_model.actions_step]

            batch_act_idx = hpg_model.generate_action_short(batch_state, batch_uids, kg_actions, seq_actions, device)
            for i in range(len(batch_uids)):
                if batch_act_idx[i] == 0:
                    pred_labels[batch_uids[i]].append(kg_actions[batch_uids[i]][step])
                elif batch_act_idx[i] == 1:
                    pred_labels[batch_uids[i]].append(seq_actions[batch_uids[i]][step])
            step += 1

        users += 64
        if users % 1280 == 0:
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "已处理:" + str(users))

    pickle.dump(pred_labels, open('./tmp/pred_labels.pkl', 'wb'))

    evaluate(pred_labels, test_set)

    print('eval finished')

def train_hpg(train_data, test_data, num_users, num_items, config):
    #load models
    args_PGPR = json.load(open('./config/pgpr.json', 'rt'))
    env = BatchKGEnvironment(args_PGPR['dataset'], args_PGPR['max_acts'], max_path_len=args_PGPR['max_path_len'],
                             state_history=args_PGPR['state_history'])
    pretrain_sd = torch.load(args_PGPR['log_dir'] + '/policy_model_epoch_50.ckpt')
    model = ActorCritic(env.state_dim, env.act_dim, gamma=args_PGPR['gamma'], hidden_sizes=args_PGPR['hidden']).to(device)
    model_sd = model.state_dict()
    model_sd.update(pretrain_sd)
    model.load_state_dict(model_sd)

    args_KERL = json.load(open('./config/kerl.json', 'rt'))
    pretrain_sd = torch.load(args_KERL['log_dir'] + '/model_epoch_35.ckpt')
    seq_model = kerl(num_users, num_items, args_KERL, device).to(device)
    model_sd = seq_model.state_dict()
    model_sd.update(pretrain_sd)
    seq_model.load_state_dict(model_sd)

    uids = list(env.kg(USER).keys())
    dataloader = ACDataLoader(uids, config.batch_size)
    #
    env_hpg = BatchHPGEnvironment(args_PGPR['dataset'], num_users, num_items, train_data, test_data, config.train_lenth)
    hpg_model = hpg(config.d*3, 2, num_items, config, device).to(device)
    optimizer = optim.Adam(hpg_model.parameters(), lr=config.lr)
    total_losses, total_plosses, total_vlosses, total_entropy, total_rewards = [], [], [], [], []
    step = 0


    data_enh = enhancement(args_PGPR['dataset'], num_items, test_data, config, device)

    # pretrain_sd = torch.load('./tmp/Amazon_Beauty/old_hpg_train' + '/policy_model_epoch_4.ckpt')
    # model_sd = hpg_model.state_dict()
    # model_sd.update(pretrain_sd)
    # hpg_model.load_state_dict(model_sd)
    #
    # pretrain_sd = torch.load('./tmp/Amazon_Beauty/old_hpg_train' + '/model_enh_epoch_4.ckpt')
    # model_sd = data_enh.state_dict()
    # model_sd.update(pretrain_sd)
    # data_enh.load_state_dict(model_sd)

    optimizer_enh = torch.optim.Adam(data_enh.parameters(), lr=config.learning_rate, weight_decay=config.l2)
    hpg_model.train()
    model.train()
    seq_model.train()
    data_enh.train()

    for epoch in range(1, config.epochs + 1):
        ### Start epoch ###
        dataloader.reset()
        while dataloader.has_next():
            batch_uids = dataloader.get_batch()  # 随机取一批用户id
            batch_state = env_hpg.user_embeds[batch_uids] #获取用户向量
            hpg_model.reset()
            env_hpg.reset(batch_uids)
            done = False
            kg_enh = [0 for uid in range(num_users)]
            seq_enh = [0 for uid in range(num_users)]

            kg_actions, seq_actions = env_hpg.batch_actions(env, model, batch_uids, device, args_PGPR['topk'],
                                                            seq_model, 'score')
            while not done:
                for uid in batch_uids:
                    if (kg_actions[uid][hpg_model.actions_step] in seq_actions[uid][:hpg_model.actions_step + 1]) and kg_enh[uid] == 0:
                        paths, probs = eh_beam_search(env, model, [uid], device, args_PGPR['topk'], kg_actions[uid][hpg_model.actions_step])
                        kg_actions_eh = env_hpg.generate_kg_action_enh(paths, probs, [uid], 'score', kg_actions[uid][:hpg_model.actions_step+1])
                        for j in range(len(kg_actions_eh[uid])):
                            kg_actions_eh[uid][j] += 1
                        if len(kg_actions[uid][hpg_model.actions_step:]) != len(kg_actions_eh[uid][:config.train_lenth-hpg_model.actions_step]):
                            print(kg_actions[uid])
                            print(kg_actions_eh[uid])
                        else:
                            prediction_score, orgin, batch_targets, Reward = data_enh.enh_train(uid
                                                                                        ,num_items
                                                                                        ,kg_actions[uid][hpg_model.actions_step:]
                                                                                        ,kg_actions_eh[uid][:config.train_lenth-hpg_model.actions_step]
                                                                                        ,config.train_lenth-hpg_model.actions_step
                                                                                        ,'kg'
                                                                                        ,test_data[uid]
                                                                                        ,device)
                            prob_enh = orgin.cpu().data.numpy()[0]
                            prob_enh = prob_enh[batch_targets.cpu().data.numpy()[0]]

                            prob = torch.from_numpy(np.array(prob_enh)).to(device)
                            prob_enh = []
                            prob_enh.append(prob)
                            prob = torch.stack(prob_enh, dim=0)

                            loss_enh = -torch.mean(torch.mul(Reward, torch.log(prob)))

                            optimizer_enh.zero_grad()
                            loss_enh.requires_grad_(True)
                            loss_enh.backward()
                            optimizer_enh.step()

                            if Reward.cpu().data.numpy()[0] * batch_targets.cpu().data.numpy()[0] == 1:
                                kg_enh[uid] = 1
                                kg_actions[uid][hpg_model.actions_step:] = kg_actions_eh[uid][:config.train_lenth - hpg_model.actions_step]


                    if (seq_actions[uid][hpg_model.actions_step] in kg_actions[uid][:hpg_model.actions_step + 1]) and seq_enh[uid] == 0:
                        seq_actions_eh = env_hpg.generate_seq_action_enh(seq_model, env_hpg.train_data, env_hpg.test_data, [uid], device, seq_actions[uid][hpg_model.actions_step], seq_actions[uid][:hpg_model.actions_step+1])

                        prediction_score, orgin, batch_targets, Reward = data_enh.enh_train(uid
                                                                                            , num_items
                                                                                            , seq_actions[uid][
                                                                                              hpg_model.actions_step:]
                                                                                            , seq_actions_eh[uid][
                                                                                              :config.train_lenth - hpg_model.actions_step]
                                                                                            ,
                                                                                            config.train_lenth - hpg_model.actions_step
                                                                                            , 'seq'
                                                                                            , test_data[uid]
                                                                                            , device)
                        prob_enh = orgin.cpu().data.numpy()[0]
                        prob_enh = prob_enh[batch_targets.cpu().data.numpy()[0]]

                        prob = torch.from_numpy(np.array(prob_enh)).to(device)
                        prob_enh = []
                        prob_enh.append(prob)
                        prob = torch.stack(prob_enh, dim=0)

                        loss_enh = -torch.mean(torch.mul(Reward, torch.log(prob)))

                        optimizer_enh.zero_grad()
                        loss_enh.requires_grad_(True)
                        loss_enh.backward()
                        optimizer_enh.step()
                        if Reward.cpu().data.numpy()[0] * batch_targets.cpu().data.numpy()[0] == 1:
                            seq_enh[uid] = 1
                            seq_actions[uid][hpg_model.actions_step:] = seq_actions_eh[uid][:config.train_lenth - hpg_model.actions_step]



                batch_act_idx = hpg_model.select_action(batch_state, batch_uids, kg_actions, seq_actions, device)
                batch_reward, done = env_hpg.batch_step(batch_uids, kg_actions, seq_actions, batch_act_idx, test_data)
                hpg_model.rewards.append(batch_reward)

            lr = config.lr * max(1e-4, 1.0 - float(step) / (config.epochs * len(uids) / config.batch_size))
            for pg in optimizer.param_groups:
                pg['lr'] = lr

             # Update policy
            total_rewards.append(np.sum(hpg_model.rewards))
            loss, ploss, vloss, eloss = hpg_model.update(optimizer, device, config.ent_weight)
            total_losses.append(loss)
            total_plosses.append(ploss)
            total_vlosses.append(vloss)
            total_entropy.append(eloss)
            step += 1

            # Report performance
            if step > 0 and step % 100 == 0:
                avg_reward = np.mean(total_rewards) / config.batch_size
                avg_loss = np.mean(total_losses)
                avg_ploss = np.mean(total_plosses)
                avg_vloss = np.mean(total_vlosses)
                avg_entropy = np.mean(total_entropy)
                total_losses, total_plosses, total_vlosses, total_entropy, total_rewards = [], [], [], [], []
                logger.info(
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
                    'epoch/step={:d}/{:d}'.format(epoch, step) +
                    ' | loss={:.5f}'.format(avg_loss) +
                    ' | ploss={:.5f}'.format(avg_ploss) +
                    ' | vloss={:.5f}'.format(avg_vloss) +
                    ' | entropy={:.5f}'.format(avg_entropy) +
                    ' | reward={:.5f}'.format(avg_reward))

        policy_file = './tmp/Amazon_Beauty/old_hpg_train/policy_model_epoch_{}.ckpt'.format(epoch)
        logger.info("Save model to " + policy_file)
        torch.save(hpg_model.state_dict(), policy_file)

        policy_file = './tmp/Amazon_Beauty/old_hpg_train/model_enh_epoch_{}.ckpt'.format(epoch)
        logger.info("Save model to " + policy_file)
        torch.save(data_enh.state_dict(), policy_file)

    print('finished')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data arguments
    parser.add_argument('--L', type=int, default=50)
    parser.add_argument('--T', type=int, default=3)
    parser.add_argument('--d', type=int, default=100)

    # train arguments
    parser.add_argument('--dataset', type=str, default=BEAUTY, help='One of {clothing, cell, beauty, cd}')
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--train_lenth', type=int, default=5)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=10, help='Max number of epochs.')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate.')
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor.')
    parser.add_argument('--ent_weight', type=float, default=1e-3, help='weight factor for entropy loss')#熵损失权重，默认0.0001
    parser.add_argument('--hidden', type=int, nargs='*', default=[512, 256], help='number of samples')

    config = parser.parse_args()

    set_random_seed(config.seed)

    data_set = Amazon.Beauty()
    train_set, test_set, num_users, num_items = data_set.generate_dataset(index_shift=1)
    valid_set, test = data_set.split_data_sequentially(test_set, test_radio=0.5)

    logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(config)

    train_hpg(train_set, valid_set, num_users, num_items, config)
    eval_hpg(train_set, valid_set, test, num_users, num_items, config)