# Comp. Engr. Bruno Brandão Soares Martins

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
from time import time, sleep
import gym
import queue
from custom_gym import CustomGym
import random
from agent import Agent



T_MAX = 10000000 # maximum steps
LEARNING_RATE = 1e-4
DISCOUNT_FACTOR = 0.99
TEST_EVERY = 30000
I_ASYNC_UPDATE = 5 #horizon for an update

training_finished = False

def async_trainer(agent, env, sess, thread_idx, T_queue, saver, save_path):

    print("Training thread ", thread_idx)
    T = T_queue.get()
    T_queue.put(T+1)
    t = 0

    last_time = time()
    last_target_update = T

    terminal = True
    while T < T_MAX:
        t_start = t
        batch_states = []
        batch_rewards = []
        batch_actions = []
        baseline_values = []

        if terminal:
            terminal = False
            state = env.reset()

        # I_ASYNC_UPDATE é a quantidade de passos que serão dados para cada update
        while not terminal and len(batch_states) < I_ASYNC_UPDATE:
            # guarda o estado atual
            batch_states.append(state)

            policy, value = agent.get_policy_and_value(state)
            action_index = np.random.choice(agent.action_size, p=policy) # escolhendo ação baseado nas probabilidades dadas pelo agente

            state, reward, terminal, info = env.step(action_index) # andando a simulação em mais um passo

            # contadores de passos (steps)
            t += 1
            T = T_queue.get()
            T_queue.put(T+1)

            reward = np.clip(reward, -1, 1)

            batch_rewards.append(reward)
            batch_actions.append(action_index)
            baseline_values.append(value[0]) # 'value' vem na forma [[value]]

        # treino dos pesos ------
        target_value = 0

        if not terminal:
            target_value = agent.get_value(state)[0]

        batch_target_values = []
        for reward in reversed(batch_rewards):
            target_value = reward + DISCOUNT_FACTOR * target_value
            batch_target_values.append(target_value)

        batch_target_values.reverse()

        batch_advantages = np.array(batch_target_values) - np.array(baseline_values)

        agent.train(np.vstack(batch_states), batch_actions, batch_target_values, batch_advantages)

    global training_finished
    training_finished = True

def a3c(game_name, num_threads=8, restore=None, save_path='./checkpoint/'):

    envs = []
    for _ in range(num_threads):
        envs.append(CustomGym(game_name))

    evaluation_env = CustomGym(game_name, record=True)

    with tf.Session() as sess:

        agent = Agent(session=sess, action_size=envs[0].action_size,
        optimizer=tf.train.AdamOptimizer(LEARNING_RATE) )

        saver = tf.train.Saver(max_to_keep=2)

        T_queue = queue.Queue()

        if restore is not None:
            saver.restore(sess, save_path + '-' + str(restore))
            last_T = restore
            print( "T was: ", last_T)
            T_queue.pu(last_T)
        else:
            sess.run(tf.global_variables_initializer())
            T_queue.put(0)

        # summary = Summary(save_path, agent)

        processes = []
        for i in range(num_threads):
            processes.append(threading.Thread(target=async_trainer, args=(agent,
            envs[i], sess, i, T_queue, saver, save_path,) ) )

        processes.append(threading.Thread(target=evaluator, args=(agent,
        evaluation_env, sess, T_queue, saver, save_path,) ) )

        for p in processes:
            p.daemon = True
            p.start()

        while not training_finished:
            sleep(0.01)

        for p in processes:
            p.join()

def test(agent, env, episodes=3):
    all_rewards = []
    all_values = []
    for i in range(episodes):
        acc_reward = 0
        state = env.reset()
        terminal = False
        while not terminal:
            policy, value = agent.get_policy_and_value(state)
            action_index = np.random.choice(agent.action_size, p=policy)
            state, reward, terminal, info = env.step(action_index)
            acc_reward += reward
            all_values.append(value)

        all_rewards.append(acc_reward)
    return all_rewards, all_values

def evaluator(agent, env, sess, T_queue, saver, save_path):
    T = T_queue.get()
    T_queue.put(T)
    last_time = time()
    last_verbose = T
    while T < T_MAX:
        T = T_queue.get()
        T_queue.put(T)
        # print("T:",T)
        T_gone = T - last_verbose
        # TEST_EVERY: testar a cada x treinos
        if T_gone >= TEST_EVERY:
            print( "T = ", T)
            current_time = time()
            print( "Train steps per second ", float(T - last_verbose) / (current_time - last_time))
            last_time = current_time
            last_verbose = T

            print("Testing agent... ")
            all_rewards, all_values = test(agent, env, episodes=3)
            avg_r = np.mean(all_rewards)
            avg_val = np.mean(all_values)
            print( "Average reward: ", avg_r, " Average value: ", avg_val)

            checkpoint_file = saver.save(sess, save_path, global_step=T)
            print( "Saved in ", checkpoint_file)
        sleep(1.0)

def main():
    a3c("Breakout-v0", num_threads=8 )

if __name__ == "__main__":
    main()
