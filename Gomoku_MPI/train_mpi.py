# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 16:16:04 2018

@author: initial-h
"""

import random
import numpy as np
import os
import shutil
import time
from mpi4py import MPI
from collections import defaultdict, deque
from game_board import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_tensorlayer import PolicyValueNet
import sys

# import sys
# sys.stdout.flush()
# or just
# mpiexec -np 43 python -u train_mpi.py

#　MPI setting
comm = MPI.COMM_WORLD
# size = comm.Get_size()
rank = comm.Get_rank()  # processing ID


class TrainPipeline():
    def __init__(self, init_model=None, transfer_model=None):
        self.game_count = 0  # count total game have played
        self.resnet_block = 19  # num of block structures in resnet
        # params of the board and the game
        self.board_width = 6
        self.board_height = 6
        self.n_in_row = 5
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # training params
        self.learn_rate = 1e-3
        self.n_playout = 200  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 500000
        # memory size, should be larger with bigger board
        # in paper it can stores 500,000 games, here with 11x11 board can store only around 2000 games
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.game_batch_num = 10000000  # total game to train

        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        # only for monitoring the progress of training
        self.pure_mcts_playout_num = 10
        # record the win rate against pure mcts
        # once the win ratio risen to 1,
        # pure mcts playout num will plus 100 and win ratio reset to 0
        
        try:
            f = open("model/win_ratio.txt", "r")
            self.best_win_ratio = float(f.read())
            f.close()
        except:
            self.best_win_ratio = -1
            
        try:
            os.remove("model/move_count.txt")
        except:
            pass
            
        # GPU setting
        # be careful to set your GPU using depends on GPUs' and CPUs' memory
        # if rank in {0,1,2}:
        #     cuda = True
        # elif rank in range(10,30):
        #     cuda = True
        #     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        #     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        # else:
        #     cuda = False

        if rank < 2:
            cuda = True
        else:
            cuda = False

        if (init_model is not None) and os.path.exists(init_model+'.index'):
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(
                self.board_width, self.board_height, block=self.resnet_block, init_model=init_model, cuda=cuda)
        elif (transfer_model is not None) and os.path.exists(transfer_model+'.index'):
            # start training from a pre-trained policy-value net
            self.policy_value_net = PolicyValueNet(
                self.board_width, self.board_height, block=self.resnet_block, transfer_model=transfer_model, cuda=cuda)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(
                self.board_width, self.board_height, block=self.resnet_block, cuda=cuda)

        self.mcts_player = MCTSPlayer(policy_value_function=self.policy_value_net.policy_value_fn_random,
                                      action_fc=self.policy_value_net.action_fc_test,
                                      evaluation_fc=self.policy_value_net.evaluation_fc2_test,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=True)

    def get_equi_data(self, play_data):
        '''
        augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        '''
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                # rotate counterclockwise 90*i
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                # np.flipud like A[::-1,...]
                # https://docs.scipy.org/doc/numpy-1.6.0/reference/generated/numpy.flipud.html
                # change the reshaped numpy
                # 0,1,2,
                # 3,4,5,
                # 6,7,8,
                # as
                # 6 7 8
                # 3 4 5
                # 0 1 2
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                # 这个np.fliplr like m[:, ::-1]
                # https://docs.scipy.org/doc/numpy/reference/generated/numpy.fliplr.html
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        '''
        collect self-play data for training
        '''
        for i in range(n_games):
            winner, play_data = self.game.start_UI_play(
                self.mcts_player, self.mcts_player, is_shown=True, rank=rank)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer_tmp.extend(play_data)
            if rank % 10 == 0:
                print('rank: {}, n_games: {}, data length: {}'.format(
                    rank, i, self.episode_len))

    def policy_update(self, print_out):
        '''
        update the policy-value net
        '''
        #play_data: [(state, mcts_prob, winner_z), ..., ...]
        # train an epoch

        tmp_buffer = np.array(self.data_buffer)
        np.random.shuffle(tmp_buffer)
        steps = len(tmp_buffer)//self.batch_size
        if print_out:
            print('tmp buffer: {}, steps: {}'.format(len(tmp_buffer), steps))
        for i in range(steps):
            mini_batch = tmp_buffer[i*self.batch_size:(i+1)*self.batch_size]
            state_batch = [data[0] for data in mini_batch]
            mcts_probs_batch = [data[1] for data in mini_batch]
            winner_batch = [data[2] for data in mini_batch]

            old_probs, old_v = self.policy_value_net.policy_value(state_batch=state_batch,
                                                                  actin_fc=self.policy_value_net.action_fc_test,
                                                                  evaluation_fc=self.policy_value_net.evaluation_fc2_test)

            loss, entropy = self.policy_value_net.train_step(state_batch,
                                                             mcts_probs_batch,
                                                             winner_batch,
                                                             self.learn_rate)

            new_probs, new_v = self.policy_value_net.policy_value(state_batch=state_batch,
                                                                  actin_fc=self.policy_value_net.action_fc_test,
                                                                  evaluation_fc=self.policy_value_net.evaluation_fc2_test)
            kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                axis=1)
            )

            explained_var_old = (1 -
                                 np.var(np.array(winner_batch) - old_v.flatten()) /
                                 np.var(np.array(winner_batch)))
            explained_var_new = (1 -
                                 np.var(np.array(winner_batch) - new_v.flatten()) /
                                 np.var(np.array(winner_batch)))

            if print_out and (steps < 10 or (i % (steps//10) == 0)):
                # print some information, not too much
                print('batch: {},length: {}'
                      'kl:{:.5f},'
                      'loss:{},'
                      'entropy:{},'
                      'explained_var_old:{:.3f},'
                      'explained_var_new:{:.3f}'.format(i,
                                                        len(mini_batch),
                                                        kl,
                                                        loss,
                                                        entropy,
                                                        explained_var_old,
                                                        explained_var_new))

        return loss, entropy

    def policy_evaluate(self, n_games=10, num=0):
        '''
        Evaluate the trained policy by
        playing against the pure MCTS player or play with itself
        pure MCTS only for monitoring the progress of training
        play with itself (last best net) for evaluating the best model so as to collect data
        '''
        # fix the playout times to 400
        current_mcts_player = MCTSPlayer(policy_value_function=self.policy_value_net.policy_value_fn_random,
                                         action_fc=self.policy_value_net.action_fc_test,
                                         evaluation_fc=self.policy_value_net.evaluation_fc2_test,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout,
                                         is_selfplay=False)
        # current_mcts_player = self.mcts_player

        test_player = MCTS_Pure(c_puct=5, n_playout=self.pure_mcts_playout_num)

        # print("---"*25, current_mcts_player.mcts._n_playout)
        # print("---"*25, test_player.mcts._n_playout)

        win_cnt = defaultdict(int)
        for i in range(n_games):
            print('rank {}: '.format(rank), 
                'Evaluating... pure mcts playout: {},  rank: {},  epoch:{}, game:{}'.format(
                self.pure_mcts_playout_num, rank, num, i)
            )

            # winner = self.game.start_play(player1=current_mcts_player,
            #                                 player2=test_player,
            #                                 start_player=i % 2,
            #                                 is_shown=False,
            #                                 print_prob=False)
                                            
            winner, _ = self.game.start_UI_play(player1=current_mcts_player, player2=test_player, is_shown=True, rank=rank)

            win_cnt[winner] += 1
            win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games # win for 1，tie for 0.5

            print('rank {}: '.format(rank), 
                'Win: {},  Lose: {},  Tie:{}, Win Ratio:{}'.format(
                win_cnt[1], win_cnt[2], win_cnt[-1], win_ratio)
            )

        # win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        
        return win_ratio, win_cnt[1], win_cnt[2], win_cnt[-1]

    def mymovefile(self, srcfile, dstfile):
        '''
        move file to another dirs
        '''
        if not os.path.isfile(srcfile):
            print("%s not exist!" % (srcfile))
        else:
            fpath, fname = os.path.split(dstfile)
            if not os.path.exists(fpath):
                os.makedirs(fpath)
            shutil.move(srcfile, dstfile)
            # print("move %s -> %s" % (srcfile, dstfile))

    def mycpfile(self, srcfile, dstfile):
        '''
        copy file to another dirs
        '''
        if not os.path.isfile(srcfile):
            print("%s not exist!" % (srcfile))
        else:
            fpath, fname = os.path.split(dstfile)
            if not os.path.exists(fpath):
                os.makedirs(fpath)
            shutil.copy(srcfile, dstfile)
            # print("move %s -> %s" % (srcfile, dstfile))

    def run(self):
        '''
        run the training pipeline
        for MPI,
        rank 0: train collected data
        rank 1: evaluate current network and save best model
        rank 2: play with pure mcts just for monitoring
        other ranks for collecting data
        '''
        # make dirs first
        if not os.path.exists('tmp'):
            os.makedirs('tmp')
        if not os.path.exists('model'):
            os.makedirs('model')

        if not os.path.exists('kifu_new'):
            os.makedirs('kifu_new')
        if not os.path.exists('kifu_train'):
            os.makedirs('kifu_train')
        if not os.path.exists('kifu_old'):
            os.makedirs('kifu_old')

        # record time for each part
        start_time = time.time()
        retore_model_time = 0
        collect_data_time = 0
        save_data_time = 0
        evaluate_time = 0

        try:
            for num in range(self.game_batch_num):
                if rank == 0:
                    # train collected data
                    before = time.time()

                    # here I move data from a dir to another in order to avoid I/O conflict
                    # it's stupid and must have a better way to do it
                    dir_kifu_new = os.listdir('kifu_new')
                    for file in dir_kifu_new:
                        if file == ".DS_Store":
                            continue
                        
                        try:
                            # try to move file from kifu_new to kifu_train, if is under written now, just pass
                            self.mymovefile('kifu_new/'+file,
                                            'kifu_train/'+file)
                        except:
                            print('rank {}: '.format(rank), 
                                '{} is being written now...'.format(file)
                            )

                    dir_kifu_train = os.listdir('kifu_train')
                    for file in dir_kifu_train:
                        if file == ".DS_Store":
                            continue
                        
                        try:
                            # load data
                            # try to move file from kifu_train to kifu_old, if is under written now, just pass
                            data = np.load('kifu_train/'+file, allow_pickle=True)
                            self.data_buffer.extend(data.tolist())
                            self.mymovefile('kifu_train/'+file, 'kifu_old/'+file)
                            self.game_count += 1
                        except:
                            print('rank {}: '.format(rank), 
                                '{} is being written now...'.format(file)
                            )

                    # print train epoch and total game num
                    print('rank {}: '.format(rank), 
                        'train epoch :{},total game :{}'.format(num, self.game_count)
                    )

                    if len(self.data_buffer) > self.batch_size * 5:
                        # training
                        print('rank {}: '.format(rank),
                            'data buffer length:{}'.format(len(self.data_buffer))
                        )
                        
                        print('rank {}: '.format(rank),
                            'Training... now time : {}'.format((time.time()-start_time)/3600)
                        )
                            
                        loss, entropy = self.policy_update(print_out=True)

                        print('rank {}: '.format(rank),
                            'Loss: {} Entropy: {}'.format(loss, entropy)
                        )

                        # save model to tmp dir, wait for evaluating
                        self.policy_value_net.save_model('tmp/current_policy.model')

                        # evaluate current model
                        evaluate_start_time = time.time()
                        win_ratio, win, lose, tie = self.policy_evaluate(n_games=10, num=num)
                        evaluate_time += time.time()-evaluate_start_time
                        if win_ratio >= self.best_win_ratio:
                            # save best model
                            print("New best policy!!!!!!!!")
                            self.best_win_ratio = win_ratio
                            self.policy_value_net.save_model('model/best_policy.model')
                            
                            f = open("model/win_ratio.txt", "w")
                            f.write(str(self.best_win_ratio) + '\n')
                            f.write(str(win) + '\n')
                            f.write(str(lose) + '\n')
                            f.write(str(tie) + '\n')
                            f.close()

                            # if (self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000):
                            #     # increase playout num and  reset the win ratio
                            #     self.pure_mcts_playout_num += 100
                            #     self.best_win_ratio = 0.0
                            # if self.pure_mcts_playout_num ==5000:
                            #     # reset mcts pure playout num
                            #     self.pure_mcts_playout_num = 1000
                            #     self.best_win_ratio = 0.0

                    after = time.time()
                    # do not train too frequent in the beginning
                    # if after-before<60*10:
                    #     time.sleep(60*10-after+before)
                    time.sleep(10)

                else:
                    #　self-play to collect data
                    if os.path.exists('tmp/current_policy.model.index'):
                        try:
                            retore_model_start_time = time.time()
                            self.policy_value_net.restore_model('tmp/current_policy.model')
                            retore_model_time += time.time()-retore_model_start_time
                            print("rank", rank, ":", 'model loaded from tmp model ...')
                        except:
                            # the model is under written
                            print("rank", rank, ":", 'cannot load model ...')

                    # tmp buffer to collect self-play data
                    self.data_buffer_tmp = []
                    # print('rank {} begin to selfplay,ronud {}'.format(rank,i+1))

                    # collect self-play data
                    collect_data_start_time = time.time()
                    self.collect_selfplay_data(self.play_batch_size)
                    collect_data_time += time.time()-collect_data_start_time

                    # save data to file
                    # it's very useful if program break off for some reason
                    # we can load the data and continue to train
                    save_data_satrt_time = time.time()
                    np.save('kifu_new/rank_'+str(rank)+'game_' + str(num)+'.npy', np.array(self.data_buffer_tmp))
                    save_data_time += time.time()-save_data_satrt_time

                    print('rank {}: '.format(rank), 'now time : {}'.format((time.time() - start_time) / 3600))
                    print('rank {}: '.format(rank), 
                        'rank : {}, restore model time : {}, collect_data_time : {}, save_data_time : {}'
                        .format(rank, retore_model_time/3600, collect_data_time/3600, save_data_time/3600)
                    )

        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    # training_pipeline = TrainPipeline(init_model='tmp/current_policy.model', transfer_model=None)
    training_pipeline = TrainPipeline(init_model=None, transfer_model='model/best_policy.model')
    # training_pipeline = TrainPipeline()
    training_pipeline.run()
