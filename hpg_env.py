from __future__ import absolute_import, division, print_function

import os
import sys
from tqdm import tqdm
import pickle
import random
import torch
from datetime import datetime


from test_agent import *
from utils import *
from interactions import *

class BatchHPGEnvironment(object):
    def __init__(self, dataset_str, num_users, num_items, train_data, test_data, max_path_len=5):
        self.embeds = load_embed(dataset_str)
        self.train_label = load_labels(dataset_str, 'train')
        self.user_embeds = self.embeds[USER]
        self.test_data = test_data
        self.train_data = Interactions(train_data, num_users, num_items)
        self.train_data.to_newsequence(50, 3)
        self.action_lenth = max_path_len

        # Following is current episode information. 以下是当前经历信息
        self._batch_path = None  # list of tuples of (relation, node_type, node_id)
        self._batch_curr_actions = None  # save current valid actions 保存当前有效动作
        self._batch_curr_state = None
        self._batch_curr_reward = None
        # Here only use 1 'done' indicator, since all paths have same length and will finish at the same time.
        self._done = False

    def generate_kg_actions(self, paths, prob, batch_uids, kg_sort):
        train_labels = self.train_label
        user_embeds = self.embeds[USER]
        purchase_embeds = self.embeds[PURCHASE][0]
        product_embeds = self.embeds[PRODUCT]
        scores = np.dot(user_embeds + purchase_embeds, product_embeds.T)

        # 1) Get all valid paths for each user, compute path score and path probability.
        pred_paths = {uid: {} for uid in batch_uids}
        for path, probs in zip(paths, prob):
            if path[-1][1] != PRODUCT:
                continue
            uid = path[0][2]
            if uid not in pred_paths:
                continue
            pid = path[-1][2]
            if pid not in pred_paths[uid]:
                pred_paths[uid][pid] = []
            path_score = scores[uid][pid]
            path_prob = reduce(lambda x, y: x * y, probs)
            pred_paths[uid][pid].append((path_score, path_prob, path))

        # 2) Pick best path for each user-product pair, also remove pid if it is in train set.
        best_pred_paths = {}
        for uid in pred_paths:
            train_pids = set(train_labels[uid])
            best_pred_paths[uid] = []
            for pid in pred_paths[uid]:
                if pid in train_pids:
                    continue
                # Get the path with highest probability
                sorted_path = sorted(pred_paths[uid][pid], key=lambda x: x[1], reverse=True)
                best_pred_paths[uid].append(sorted_path[0])

        # 3) Compute top 5 recommended products for each user.
        sort_by = kg_sort
        pred_labels = {}
        for uid in best_pred_paths:
            if sort_by == 'score':
                sorted_path = sorted(best_pred_paths[uid], key=lambda x: (x[0], x[1]), reverse=True)
            elif sort_by == 'prob':
                sorted_path = sorted(best_pred_paths[uid], key=lambda x: (x[1], x[0]), reverse=True)
            top5_pids = [p[-1][2] for _, _, p in sorted_path[:5]]  # from largest to smallest
            # add up to 5 pids if not enough
            if len(top5_pids) < 5:
                train_pids = set(train_labels[uid])
                cand_pids = np.argsort(scores[uid])
                for cand_pid in cand_pids[::-1]:
                    if cand_pid in train_pids or cand_pid in top5_pids:
                        continue
                    top5_pids.append(cand_pid)
                    if len(top5_pids) >= 5:
                        break
            # end of add
            pred_labels[uid] = top5_pids[::-1]
        return pred_labels


    def generate_seq_actions(self, kerl, train, test_set, batch_uids, device):
        num_users = train.num_users
        num_items = train.num_items
        user_indexes = np.arange(num_users)
        item_indexes = np.arange(num_items)
        pred_list = None
        test_sequences = train.test_sequences.sequences
        test_len = train.test_sequences.length

        batch_test = []
        for i in batch_uids:
            batch_test.append(test_set[i])

        batch_user_index = user_indexes[batch_uids]

        batch_test_sequence = test_sequences[batch_user_index]
        batch_test_sequence = np.atleast_2d(batch_test_sequence)
        batch_test_lenth = test_len[batch_user_index]

        batch_test_len = torch.from_numpy(batch_test_lenth).type(torch.LongTensor).to(device)
        batch_test_sequences = torch.from_numpy(batch_test_sequence).type(torch.LongTensor).to(device)

        all_top5 = {}
        for i in range(5):
            prediction_score = kerl(batch_test_sequences, batch_test_len)
            rating_pred = prediction_score
            rating_pred = rating_pred.cpu().data.numpy().copy()
            pred_list = rating_pred
            if i == 0:
                for uid in batch_uids:
                    all_top5[uid] = []
            for x, each_policy in enumerate(pred_list[:, 0, :]):
                each_sample = -each_policy
                for one_seq_action in np.argsort(each_sample):
                    if item_indexes[one_seq_action] != 0 and item_indexes[one_seq_action] not in all_top5[batch_uids[x]]:
                        break
                one_item = item_indexes[one_seq_action]
                all_top5[batch_uids[x]].append(one_item)
                if batch_test_lenth[x] < 50:
                    batch_test_sequence[x][batch_test_lenth[x]] = one_item
                    batch_test_lenth[x] += 1
                elif batch_test_lenth[x] == 50:
                    for y in range(49):
                        batch_test_sequence[x][y] = batch_test_sequence[x][y+1]
                    batch_test_sequence[x][49] = one_item
            batch_test_sequences = torch.from_numpy(batch_test_sequence).type(torch.LongTensor).to(device)
            batch_test_len = torch.from_numpy(batch_test_lenth).type(torch.LongTensor).to(device)
        return all_top5

    def generate_seq_action(self, kerl, train, test_set, batch_uids, device):
        num_users = train.num_users
        num_items = train.num_items
        user_indexes = np.arange(num_users)
        item_indexes = np.arange(num_items)
        pred_list = None
        test_sequences = train.test_sequences.sequences
        test_len = train.test_sequences.length

        batch_test = []
        for i in batch_uids:
            batch_test.append(test_set[i])

        batch_user_index = user_indexes[batch_uids]

        batch_test_sequence = test_sequences[batch_user_index]
        batch_test_sequence = np.atleast_2d(batch_test_sequence)
        batch_test_lenth = test_len[batch_user_index]

        batch_test_len = torch.from_numpy(batch_test_lenth).type(torch.LongTensor).to(device)
        batch_test_sequences = torch.from_numpy(batch_test_sequence).type(torch.LongTensor).to(device)

        all_top5 = {}
        for uid in batch_uids:
            all_top5[uid] = []

        prediction_score = kerl(batch_test_sequences, batch_test_len)
        rating_pred = prediction_score
        rating_pred = rating_pred.cpu().data.numpy().copy()
        pred_list = rating_pred
        for x, each_policy in enumerate(pred_list[:, 0, :]):
            each_sample = -each_policy
            for i in range(5):
                for one_seq_action in np.argsort(each_sample):
                    if item_indexes[one_seq_action] != 0 and item_indexes[one_seq_action] not in all_top5[batch_uids[x]]:
                        break
                one_item = item_indexes[one_seq_action]
                all_top5[batch_uids[x]].append(one_item)

        return all_top5

    def generate_seq_action_enh(self, kerl, train, test_set, batch_uids, device, enh_product, seq_actions):
        num_users = train.num_users
        num_items = train.num_items
        user_indexes = np.arange(num_users)
        item_indexes = np.arange(num_items)
        pred_list = None
        test_sequences = train.test_sequences.sequences
        test_len = train.test_sequences.length

        batch_test = []
        for i in batch_uids:
            batch_test.append(test_set[i])

        batch_user_index = user_indexes[batch_uids]

        batch_test_lenth = test_len[batch_user_index]
        batch_test_sequence = test_sequences[batch_user_index]
        if batch_test_lenth[0] < 50:
            batch_test_sequence[0][batch_test_lenth[0]] = enh_product
        elif batch_test_lenth[0] == 50:
            for i in range(49):
                batch_test_sequence[0][i] = batch_test_sequence[0][i+1]
            batch_test_lenth[0] -= 1
            batch_test_sequence[0][batch_test_lenth[0]] = enh_product

        batch_test_sequence = np.atleast_2d(batch_test_sequence)
        batch_test_lenth[0] += 1


        batch_test_len = torch.from_numpy(batch_test_lenth).type(torch.LongTensor).to(device)
        batch_test_sequences = torch.from_numpy(batch_test_sequence).type(torch.LongTensor).to(device)

        all_top5 = {}
        for uid in batch_uids:
            all_top5[uid] = []

        prediction_score = kerl(batch_test_sequences, batch_test_len)

        rating_pred = np.squeeze(prediction_score, axis=-1)
        rating_preds = []
        rating_preds.append(rating_pred)
        rating_pred = torch.stack(rating_preds, dim=0)
        rating_pred = rating_pred.cpu().data.numpy().copy()

        pred_list = rating_pred
        # for x, each_policy in enumerate(pred_list[:, :]):
        #     each_sample = -each_policy
        #     for i in range(5):
        #         for one_seq_action in np.argsort(each_sample):
        #             if item_indexes[one_seq_action] != 0 and item_indexes[one_seq_action] not in all_top5[batch_uids[x]]:
        #                 break
        #         one_item = item_indexes[one_seq_action]
        #         all_top5[batch_uids[x]].append(one_item)
        # print(enh_product)
        # print(seq_actions)
        # print(all_top5)
        # all_top5[batch_uids[0]] = []
        for x, each_policy in enumerate(pred_list[:, :]):
            each_sample = -each_policy
            for i in range(5):
                for one_seq_action in np.argsort(each_sample):
                    if (item_indexes[one_seq_action] != 0) and (item_indexes[one_seq_action] not in all_top5[batch_uids[x]]) and (item_indexes[one_seq_action] not in seq_actions):
                        break
                one_item = item_indexes[one_seq_action]
                all_top5[batch_uids[x]].append(one_item)
        # print(all_top5)

        return all_top5

    def generate_kg_action_enh(self, paths, prob, batch_uids, kg_sort, kg_actions):
        train_labels = self.train_label
        user_embeds = self.embeds[USER]
        purchase_embeds = self.embeds[PURCHASE][0]
        product_embeds = self.embeds[PRODUCT]
        scores = np.dot(user_embeds + purchase_embeds, product_embeds.T)

        # 1) Get all valid paths for each user, compute path score and path probability.
        pred_paths = {uid: {} for uid in batch_uids}
        for path, probs in zip(paths, prob):
            if path[-1][1] != PRODUCT:
                continue
            uid = path[0][2]
            if uid not in pred_paths:
                continue
            pid = path[-1][2]
            if pid not in pred_paths[uid]:
                pred_paths[uid][pid] = []
            path_score = scores[uid][pid]
            path_prob = reduce(lambda x, y: x * y, probs)
            pred_paths[uid][pid].append((path_score, path_prob, path))

        # 2) Pick best path for each user-product pair, also remove pid if it is in train set.
        best_pred_paths = {}
        for uid in pred_paths:
            train_pids = set(train_labels[uid])
            best_pred_paths[uid] = []
            for pid in pred_paths[uid]:
                if pid in train_pids:
                    continue
                # Get the path with highest probability
                sorted_path = sorted(pred_paths[uid][pid], key=lambda x: x[1], reverse=True)
                best_pred_paths[uid].append(sorted_path[0])

        # 3) Compute top 5 recommended products for each user.
        sort_by = kg_sort
        pred_labels = {}
        for uid in best_pred_paths:
            if sort_by == 'score':
                sorted_path = sorted(best_pred_paths[uid], key=lambda x: (x[0], x[1]), reverse=True)
            elif sort_by == 'prob':
                sorted_path = sorted(best_pred_paths[uid], key=lambda x: (x[1], x[0]), reverse=True)

            top5_pids = [p[-1][2] for _, _, p in sorted_path[:5]]  # from largest to smallest
            # add up to 5 pids if not enough
            if len(top5_pids) < 5:
                train_pids = set(train_labels[uid])
                cand_pids = np.argsort(scores[uid])
                for cand_pid in cand_pids[::-1]:
                    if cand_pid in train_pids or cand_pid in top5_pids:
                        continue
                    top5_pids.append(cand_pid)
                    if len(top5_pids) >= 5:
                        break
            top5_pids_old = top5_pids

            top5_pids = []
            is_replace = 0
            for _, _, p in sorted_path[:20]:
                one_item = p[-1][2]
                if one_item+1 not in kg_actions:
                    top5_pids.append(one_item)
                    if len(top5_pids) == 5:
                        break
                # elif is_replace == 0:
                #     is_replace = 1
                #     print(sorted_path)
                #     print(kg_actions)
                #     print(top5_pids_old)
            # end of add

            pred_labels[uid] = top5_pids[::-1]
            # if is_replace == 1:
            #     print(top5_pids)
            #     print(pred_labels)
        return pred_labels


    def batch_actions(self, env, model, batch_uids, device, topk, seq_model, kg_sort = 'prob'):
        paths, probs = batch_beam_search(env, model, batch_uids, device, topk)
        kg_actions = self.generate_kg_actions(paths, probs, batch_uids, kg_sort)
        seq_actions = self.generate_seq_action(seq_model, self.train_data, self.test_data, batch_uids, device)
        for i in batch_uids:
            for j in range(len(kg_actions[i])):
                kg_actions[i][j] += 1

        return kg_actions, seq_actions

    def _is_done(self):
        """Episode ends only if max path length is reached."""
        return self._done or len(self._batch_path[0]) >= self.action_lenth


    def _batch_get_reward(self, batch_uids, kg_actions, seq_actions, batch_path, test_data):
        batch_reward = []
        for i in range(len(batch_uids)):
            if batch_path[i][-1] == 0 :
                if kg_actions[batch_uids[i]][len(batch_path[i])-1] in test_data[batch_uids[i]]:
                    batch_reward.append(1.0)
                else:
                    batch_reward.append(0.0)
            elif batch_path[i][-1] == 1 :
                if seq_actions[batch_uids[i]][len(batch_path[i])-1] in test_data[batch_uids[i]]:
                    batch_reward.append(1.0)
                else:
                    batch_reward.append(0.0)
        return np.array(batch_reward)

    def batch_step(self, batch_uids, kg_actions, seq_actions, batch_act_idx, test_data):
        """
        Args:
            batch_act_idx: list of integers.
        Returns:
            batch_next_state: numpy array of size [bs, state_dim].
            batch_reward: numpy array of size [bs].
            done: True/False
        """
        assert len(batch_act_idx) == len(self._batch_path)

        # Execute batch actions.
        for i in range(len(batch_act_idx)):
            act_idx = batch_act_idx[i]
            self._batch_path[i].append(act_idx)

        self._done = self._is_done()  # must run before get actions, etc.
        # self._batch_curr_state = self._batch_get_state(self._batch_path)
        # self._batch_curr_actions = self._batch_get_actions(self._batch_path, self._done)
        self._batch_curr_reward = self._batch_get_reward(batch_uids, kg_actions, seq_actions, self._batch_path, test_data)

        return self._batch_curr_reward, self._done

    def _batch_get_rewards(self, batch_uids, kg_actions, seq_actions, batch_path, test_data, probs):
        batch_reward = []
        batch_prob = probs.cpu().detach().numpy().tolist()
        for i in range(len(batch_uids)):
            if batch_path[i][-1] == 0:
                if kg_actions[i] in test_data[batch_uids[i]]:
                    batch_reward.append(1.0 / batch_prob[i][0])
                else:
                    batch_reward.append(0.0)
            elif batch_path[i][-1] == 1:
                if seq_actions[i] in test_data[batch_uids[i]]:
                    batch_reward.append(1.0 / batch_prob[i][1])
                else:
                    batch_reward.append(0.0)
        return np.array(batch_reward)

    def batch_steps(self, batch_uids, kg_actions, seq_actions, batch_act_idx, test_data, probs):
        """
        Args:
            batch_act_idx: list of integers.
        Returns:
            batch_next_state: numpy array of size [bs, state_dim].
            batch_reward: numpy array of size [bs].
            done: True/False
        """
        assert len(batch_act_idx) == len(self._batch_path)

        # Execute batch actions.
        for i in range(len(batch_act_idx)):
            act_idx = batch_act_idx[i]
            self._batch_path[i].append(act_idx)

        self._done = self._is_done()  # must run before get actions, etc.
        # self._batch_curr_state = self._batch_get_state(self._batch_path)
        # self._batch_curr_actions = self._batch_get_actions(self._batch_path, self._done)
        self._batch_curr_reward = self._batch_get_rewards(batch_uids, kg_actions, seq_actions, self._batch_path, test_data, probs)

        return self._batch_curr_reward, self._done

    def reset(self, uids=None):
        if uids is None:
            all_uids = list(self.kg(USER).keys())
            uids = [random.choice(all_uids)]

        # each element is a tuple of (relation, entity_type, entity_id)
        self._batch_path = [[] for uid in uids]
        self._done = False

        return
