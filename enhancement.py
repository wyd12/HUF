import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from eval_metrics import *
from model.DynamicGRU import DynamicGRU
from test_agent import *
from utils import *

class enhancement(nn.Module):
    def __init__(self, dataset_str, num_items, test_data, config, device):
        super(enhancement, self).__init__()
        self.embeds = load_embed(dataset_str)
        self.user_embeds = self.embeds[USER]
        self.test_data = test_data
        self.item_embeddings = nn.Embedding(num_items, config.n_iter).to(device)
        self.config = config
        self.device = device

        self.lamda = 10
        dims = self.config.n_iter

        self.DP = nn.Dropout(0.5)
        self.seq_enc = DynamicGRU(input_dim=dims, output_dim=dims, bidirectional=False, batch_first=True)
        self.kg_enc = DynamicGRU(input_dim=dims, output_dim=dims, bidirectional=False, batch_first=True)

        self.mlp = nn.Linear(dims*3, dims*2)
        self.fc = nn.Linear(dims*2, 2)

    def forward(self, batch_sequences, train_len):
        #test process
        probs = []
        input = self.item_embeddings(batch_sequences)
        out_enc, h = self.enc(input, train_len)

        # kg_map = self.BN(self.kg_map)
        # kg_map =kg_map.detach()
        # batch_kg = self.get_kg(batch_sequences,train_len,kg_map)

        # mlp_in = torch.cat([h.squeeze(),batch_kg, self.mlp_history(batch_kg)],dim=1)
        mlp_in = h.squeeze()
        mlp_hidden = self.mlp(mlp_in)
        mlp_hidden = torch.tanh(mlp_hidden)

        out = self.fc(mlp_hidden)
        probs.append(out)
        return torch.stack(probs, dim=1)

    def enh_train(self, uid, num_items, sequence, sequence_enh, trainlen, seq_or_kg, target, device):
        probs = []
        probs_orgin = []
        each_sample = []
        Rewards = []
        if len(sequence) != len(sequence_enh):
            print(uid)
            print(sequence)
            print(sequence_enh)
            print(trainlen)
            print('*****************')

        sequences = np.asarray([sequence])
        sequences_enh = np.asarray([sequence_enh])
        train_len = np.asarray([trainlen])

        sequences = torch.from_numpy(sequences).type(torch.LongTensor).to(device)
        sequences_enh = torch.from_numpy(sequences_enh).type(torch.LongTensor).to(device)
        train_len = torch.from_numpy(train_len).type(torch.LongTensor).to(device)
        pred_one_hot = np.zeros((len([uid]), num_items))
        batch_tar = target
        for i, tar in enumerate(batch_tar):
            pred_one_hot[0][tar] = 0.2
        pred_one_hot = torch.from_numpy(pred_one_hot).type(torch.FloatTensor).to(device)

        input = self.item_embeddings(sequences)
        input_enh = self.item_embeddings(sequences_enh)
        if seq_or_kg == 'kg':
            self.kg_enc.to(device)
            out_enc, h = self.kg_enc(input, train_len)
            out_enc_enh, h_enh = self.kg_enc(input_enh, train_len)
        elif seq_or_kg == 'seq':
            self.seq_enc.to(device)
            out_enc, h = self.seq_enc(input, train_len)
            out_enc_enh, h_enh = self.seq_enc(input_enh, train_len)

        embed_user = torch.from_numpy(self.embeds[USER][uid]).to(device)
        mlp_in = torch.cat([embed_user, h.squeeze(), h_enh.squeeze()], dim=0)

        self.mlp.to(device)
        mlp_hidden = self.mlp(mlp_in)
        mlp_hidden = torch.tanh(mlp_hidden)
        self.fc.to(device)
        out_fc = self.fc(mlp_hidden)


        '''
        When sampling episodes, we increased the probability of ground truth to improve the convergence efficiency
        '''
        out_distribution = F.softmax(out_fc, dim=0)
        probs_orgin.append(out_distribution)
        # pai-->p(a|s)
        probs.append(out_distribution)
        m = torch.distributions.categorical.Categorical(out_distribution)
        # action
        sample1 = m.sample()
        each_sample.append(sample1)
        sample = each_sample[0].cpu().data.numpy()
        # generate 3 episode
        # Reward = self.generateReward(sample1, self.args.T-1, 3, target, pred_one_hot, h, train_len)
        # Rewards.append(Reward)
        intersection = list(set(sequence) & set(target))
        intersection_enh = list(set(sequence_enh) & set(target))

        if len(intersection_enh) > len(intersection) and sample == 1:
            Rewards.append(1)
        elif len(intersection_enh) <= len(intersection) and sample == 0:
            Rewards.append(1)
        else:
            Rewards.append(0)
        # dec_input_target = self.item_embeddings(items_to_predict)


        Reward = torch.from_numpy(np.array(Rewards[0])).to(device)
        Rewards = []
        Rewards.append(Reward)

        probs = torch.stack(probs, dim=0)
        probs_orgin = torch.stack(probs_orgin, dim=0)
        each_sample = torch.stack(each_sample, dim=0)
        Rewards = torch.stack(Rewards, dim=0)
        return probs, probs_orgin, each_sample, Rewards

    # def get_kg(self,batch_sequences,trainlen,kg_map):
    #     # batch_kg_avg
    #     batch_kg = []
    #     for i, seq in enumerate(batch_sequences):
    #         seq_kg = kg_map[seq]
    #         seq_kg_avg = torch.sum(seq_kg,dim=0)
    #         seq_kg_avg = torch.div(seq_kg_avg,trainlen[i])
    #         batch_kg.append(seq_kg_avg)
    #     batch_kg = torch.stack(batch_kg)
    #     return batch_kg

    def generateReward(self, sample1, path_len, path_num, items_to_predict, pred_one_hot,h_orin,tarlen):
        # history_kg = self.mlp_history(batch_kg)
        Reward = []
        dist = []
        dist_replay = []
        for paths in range(path_num):
            h = h_orin
            indexes = []
            indexes.append(sample1)
            dec_inp_index = sample1
            dec_inp = self.item_embeddings(dec_inp_index)
            dec_inp = dec_inp.unsqueeze(1)
            # ground_kg = self.get_kg(items_to_predict[:, self.args.T - path_len - 1:],tarlen,kg_map)
            for i in range(path_len):
                out_enc, h = self.enc(dec_inp, h, one=True)
                # out_fc = self.fc(h.squeeze())
                # mlp_in = torch.cat([h.squeeze(), batch_kg, self.mlp_history(batch_kg)], dim=1)
                mlp_in = h.squeeze()
                mlp_hidden = self.mlp(mlp_in)
                mlp_hidden = torch.tanh(mlp_hidden)
                out_fc = self.fc(mlp_hidden)

                out_distribution = F.softmax(out_fc, dim=1)
                out_distribution = 0.8 * out_distribution
                out_distribution = torch.add(out_distribution, pred_one_hot)
                # pai-->p(a|s)
                m = torch.distributions.categorical.Categorical(out_distribution)
                sample2 = m.sample()
                dec_inp = self.item_embeddings(sample2)
                dec_inp = dec_inp.unsqueeze(1)
                indexes.append(sample2)
            indexes = torch.stack(indexes, dim=1)
            # episode_kg = self.get_kg(indexes,torch.Tensor([path_len+1]*len(indexes)),kg_map)

            '''
            dist: knowledge reward
            dist_replay: induction network training (rank)
            '''
            # dist.append(self.cos(episode_kg ,ground_kg))
            # dist_replay.append(self.cos(episode_kg,history_kg))
            #Reward.append(bleu_each(indexes,items_to_predict[:,self.args.T-path_len-1:]))
            Reward.append(dcg_k(items_to_predict[:, self.args.T - path_len - 1:], indexes, path_len + 1))
        Reward = torch.FloatTensor(Reward).to(self.device)
        # dist = torch.stack(dist, dim=0)
        # dist = torch.mean(dist, dim=0)

        # dist_replay = torch.stack(dist_replay, dim=0)
        # dist_sort = self.compare_kgReawrd(Reward, dist_replay)
        Reward = torch.mean(Reward, dim=0)
        # Reward = Reward + self.lamda * dist
        Reward = Reward
        # dist.sort =dist_sort.detach()
        return Reward


    def compare_kgReawrd(self, reward, dist):
        logit_reward, indice = reward.sort(dim=0)
        dist_sort = dist.gather(dim=0, index=indice)
        return dist_sort

