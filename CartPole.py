import random
import math
import numpy as np
from collections import deque
import gym
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam

#################################################
env = gym.make('CartPole-v0')

'''for i in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        
        if done:
            print("episode fished after {} timesteps".format(t+1))
            break'''

print(env.action_space)
print(env.observation_space)

print(env.observation_space.high)
print(env.observation_space.low)

#############################################################################
# Define our model
def OurModel():
    model = Sequential()
    # Input layer of state size
    # Hidden Layer with 64 nodes and the activation function is relu
    model.add(Dense(64, input_dim = 4, activation = 'relu'))
   
    # Hidden Layer with 64 nodes and the activation function is relu
    model.add(Dense(64, activation = 'relu'))
    
    # Hidden Layer with 32 nodes and the activation function is relu
    model.add(Dense(32, activation = 'relu'))
     # Hidden Layer with 32 nodes and the activation function is relu
    model.add(Dense(32, activation = 'relu'))
    
    # Output layer with two nodes (action_space: left and right) 
    # linear activation function
    model.add(Dense(2, activation = 'linear')) 
    
    model.compile(loss = 'mse', optimizer = Adam(learning_rate= 0.01, decay=0.01))
    model.summary()
    return model
    
#############################################################################
class OUR_DQN:
    def __init__(self):
        
        #Define parameters
        self.num_episodes = 1000        #number of epoch
        self.num_win_ticks = 195       
        self.max_env_steps = None

        self.gamma = 1.0               # Discount Factor #reward
        self.epsilon = 1.0              # always random steps (Exploration)

        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999

       

        self.batch_size = 32

        self.monitor = False
        self.quiet = False

        # Environment Parameters
        self.memory = deque(maxlen = 100000)
        self.env = gym.make('CartPole-v0')
        # Create Our model
        self.model=OurModel()
        
    ##################################################### Defining necessary functions
    
    def experience(self, state, action, reward, next_state, done):
         self.memory.append((state, action, reward, next_state, done))
         if self.epsilon > self.epsilon_min :
            self.epsilon *= self.epsilon_decay
          
          
    def get_action(self, state):
       if (np.random.random() <= self.epsilon):
         return self.env.action_space.sample()
       else:
         return np.argmax(self.model.predict(state))
     
    def preprocess_state(self, state):
        return np.reshape(state,[1,4])
    
    
    def replay(self):
         x_batch, y_batch = [], []
         minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
    
         for state, action, reward, next_state, done in minibatch:
            y_target =self.model.predict(state)
            
            if done:
                y_target[0][action] = reward
            else:
                y_target[0][action] = reward + self.gamma * np.max(self.model.predict(next_state)[0])
                x_batch.append(state[0])
                y_batch.append(y_target[0])
         # Train the Neural Network with batches  
         self.model.fit(np.array(x_batch), np.array(y_batch),batch_size=len(x_batch), verbose=0)
    #######################################################################
    #Define run function
    def run(self):
       scores = deque(maxlen=100)
    
       for e in range(self.num_episodes ):
           state = self.preprocess_state(self.env.reset())
           done = False
           i = 0
           
           while not done:
              action = self.get_action(state)
              next_state, reward, done, _ = self.env.step(action)
              self.env.render()
              next_state = self.preprocess_state(next_state)
              
              
              self.experience(state, action, reward, next_state, done)
              state = next_state
              i += 1
          
           scores.append(i)
           mean_score = np.mean(scores)
           
           
           if mean_score >= self.num_win_ticks and e >= 100:
              if not self.quiet:
                   print('Ran {} Episodes , Solvied after {} trails'.format(e, e-100))
                
              return 
           if e % 5 == 0 and not self.quiet:
              print('[Episode {}] - Mean Survival time over last 100 episoded was {} ticks .' .format(e, mean_score))
            
           self.replay()
           print("Saving trained model as cartpole-dqn.h5")
           self.save("cartpole-dqn.h5")
       if not self.quiet:
        print ('Did not solve after {} episodes '.format(e))
        
       return e
    ####################################################################
    def test(self):
        self.load("cartpole-dqn.h5")
        for e in range(self.num_episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, 4])
            done = False
            i = 0
            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, 4])
                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.num_episodes, i))
                    break
               
    
    #####################################################################
    # load and save function:
    
    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)
    #######################################################################
    
if __name__ == "__main__":
    agent = OUR_DQN()
    #agent.run()
    agent.test()   