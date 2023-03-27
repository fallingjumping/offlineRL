import numpy as np

class env():
    def __init__(self):
        self.state_dim = 9
        self.action_dim = 1
        # self.batch = np.zeros((self.batch_size,self.state_dim))
        self.action = np.zeros(self.action_dim)
        self.state = np.zeros(self.state_dim)
        self.action_bound = []
        self.goal_moisture = 13.55
        # self.reward = 0
        self.max_action = 133.89
        
    def step(self): # ,action
        self.state, self.action, notdone, next_state = self.data.state_action()
        # print(self.state,self.action)
        moisture = next_state[0][6]
        reward = (-100) * (abs(moisture - self.goal_moisture))
        # reward = -(1/2) + 1/(1 + math.exp(abs(moisture - self.goal_moisture)))
        # print(reward)
        if notdone == 0:
            if reward < 0:
                reward = 1
        # self.state = next_state
        return self.state, self.action, reward, notdone, next_state

    def reset(self):
        self.data.get_data()
        self.state, _, _, _ = self.data.state_action()
        return self.state
