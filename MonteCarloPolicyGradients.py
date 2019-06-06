#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 22:21:36 2019

@author: minghao
"""

import tensorflow as tf
import numpy as np
import gym

env = gym.make('CartPole-v0')
env = env.unwrapped
env.seed(1)

state_size = 4
action_size = env.action_space.n

max_episodes = 3000
learning_rate = 0.01
gamma = 0.95

def discount_and_normalize_rewards(episode_rewards):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative*gamma+episode_rewards[i]
        discounted_episode_rewards[i] = cumulative
    
    discounted_episode_rewards = discounted_episode_rewards[::-1].cumsum()[::-1]
    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards-mean)/std
    
    return discounted_episode_rewards

with tf.name_scope("inputs"):
    input_ = tf.placeholder(tf.float32, [None, state_size], name="input_")
    actions = tf.placeholder(tf.int32, [None, action_size], name="actions")
    discounted_episode_rewards_ = tf.placeholder(tf.float32, [None,], name="discounted_episode_rewards")
    
    # Add this placeholder for having this variable in tensorboard
    mean_reward_ = tf.placeholder(tf.float32 , name="mean_reward")

    with tf.name_scope("fc1"):
        fc1 = tf.layers.dense(inputs = input_,
                                               units = 10,
                                                activation =tf.nn.relu,
                                                kernel_initializer=tf.glorot_normal_initializer())

    with tf.name_scope("fc2"):
        fc2 = tf.layers.dense(inputs = fc1,
                                                units = action_size,
                                                activation = tf.nn.relu,
                                                kernel_initializer=tf.glorot_normal_initializer())
    
    with tf.name_scope("fc3"):
        fc3 = tf.layers.dense(inputs = fc2,
                                                units = action_size,
                                                activation = None,
                                               kernel_initializer=tf.glorot_normal_initializer())

    with tf.name_scope("softmax"):
        action_distribution = tf.nn.softmax(fc3)

    with tf.name_scope("loss"):
        # tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the softmax function
        # If you have single-class labels, where an object can only belong to one class, you might now consider using 
        # tf.nn.sparse_softmax_cross_entropy_with_logits so that you don't have to convert your labels to a dense one-hot array. 
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = fc3, labels = actions)
        loss = tf.reduce_mean(neg_log_prob * discounted_episode_rewards_) 
        
    
    with tf.name_scope("train"):
        train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        
# Setup TensorBoard Writer
writer = tf.summary.FileWriter("/home/minghao/tensorboard/pg/1")

## Losses
tf.summary.scalar("Loss", loss)

## Reward mean
tf.summary.scalar("Reward_mean", mean_reward_)

write_op = tf.summary.merge_all()

allRewards = []
total_rewards = 0
maximumRewardRecorded = 0
episode = 0
episode_states, episode_actions, episode_rewards = [],[],[]

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for episode in range(max_episodes):
        
        episode_rewards_sum = 0

        # Launch the game
        state = env.reset()
        
        env.render()
           
        while True:
            
            # Choose action a, remember WE'RE NOT IN A DETERMINISTIC ENVIRONMENT, WE'RE OUTPUT PROBABILITIES.
            action_probability_distribution = sess.run(action_distribution, feed_dict={input_: state.reshape([1,4])})
            
            action = np.random.choice(range(action_probability_distribution.shape[1]), p=action_probability_distribution.ravel())  # select action w.r.t the actions prob

            # Perform a
            new_state, reward, done, info = env.step(action)

            # Store s, a, r
            episode_states.append(state)
                        
            # For actions because we output only one (the index) we need 2 (1 is for the action taken)
            # We need [0., 1.] (if we take right) not just the index
            action_ = np.zeros(action_size)
            action_[action] = 1
            
            episode_actions.append(action_)
            
            episode_rewards.append(reward)
            if done:
                # Calculate sum reward
                episode_rewards_sum = np.sum(episode_rewards)
                
                allRewards.append(episode_rewards_sum)
                
                total_rewards = np.sum(allRewards)
                
                # Mean reward
                mean_reward = np.divide(total_rewards, episode+1)
                
                
                maximumRewardRecorded = np.amax(allRewards)
                
                print("==========================================")
                print("Episode: ", episode)
                print("Reward: ", episode_rewards_sum)
                print("Mean Reward", mean_reward)
                print("Max reward so far: ", maximumRewardRecorded)
                
                # Calculate discounted reward
                discounted_episode_rewards = discount_and_normalize_rewards(episode_rewards)
            
                                
                # Feedforward, gradient and backpropagation
                loss_, _ = sess.run([loss, train_opt], feed_dict={input_: np.vstack(np.array(episode_states)),
                                                                 actions: np.vstack(np.array(episode_actions)),
                                                                 discounted_episode_rewards_: discounted_episode_rewards 
                                                                })
                
 
                                                                 
                # Write TF Summaries
                summary = sess.run(write_op, feed_dict={input_: np.vstack(np.array(episode_states)),
                                                                 actions: np.vstack(np.array(episode_actions)),
                                                                 discounted_episode_rewards_: discounted_episode_rewards,
                                                                    mean_reward_: mean_reward
                                                                })
                
               
                writer.add_summary(summary, episode)
                writer.flush()
                
            
                
                # Reset the transition stores
                episode_states, episode_actions, episode_rewards = [],[],[]
                
                break
            
            state = new_state
        
        # Save Model
        if episode % 100 == 0:
            saver.save(sess, "./models/model.ckpt")
            print("Model saved")
            
            
            
import time
with tf.Session() as sess:
    env.reset()
    rewards = []
    
    # Load the model
    saver.restore(sess, "./models/model.ckpt")

    for episode in range(50):
        state = env.reset()
        step = 0
        done = False
        total_rewards = 0
        print("****************************************************")
        print("EPISODE ", episode)

        while True:
            

            # Choose action a, remember WE'RE NOT IN A DETERMINISTIC ENVIRONMENT, WE'RE OUTPUT PROBABILITIES.
            action_probability_distribution = sess.run(action_distribution, feed_dict={input_: state.reshape([1,4])})
            #print(action_probability_distribution)
            action = np.random.choice(range(action_probability_distribution.shape[1]), p=action_probability_distribution.ravel())  # select action w.r.t the actions prob


            new_state, reward, done, info = env.step(action)

            total_rewards += reward

            if done:
                rewards.append(total_rewards)
                print ("Score", total_rewards)
                break
            state = new_state
            time.sleep(0.01)
            env.render()
        
    env.close()
    print ("Score over time: " +  str(sum(rewards)/10))