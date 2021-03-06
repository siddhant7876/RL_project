# -*- coding: utf-8 -*-
"""EvolGame_by_conceited_altruists.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1z4xKteuubfgQPKHHmewvQaywMLNw9VU0

# **Model Explaination**

We have four types of strategy : 


*   Always Cooperate (AC) : Has the maximum probability of sharing food and getting food in return.
*   Tit For Tat (TFT) : Decides based on the history of the agent at mercy.
*   Alternatively cooperate and defect (ALT) : Alternates between cooperate and defect.
*   Always Defect (AD) : Least willing to share food.
"""

#!pip install pygame

from email.policy import Policy
from select import select
import numpy as np
import matplotlib.pyplot as plt
import random
import pygame
import dqn
import math
from traitlets import Float
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

strat_names = ['AC','TFT','ALT','AD','Intel']
basket_of_strat = {'AC':[0.8],'TFT':[],'ALT':[0.6,0.4],'AD':[0.2],'Intel':[]}
num_to_strat = {0:'AC',1:'TFT',2:'ALT',3:'AD',4:'Intel'}
strat_to_num = {'AC':0,'TFT':1,'ALT':2,'AD':3,'Intel':4}
gridcolor = (255,128,255)
# pop_of_strat = {'AC':[0],'TFT':[0],'ALT':[0],'AD':[0],'Intel':[0]}
# self.pop_of_size = {1:[0],2:[0]}
foodimg = pygame.image.load('/home/siddhss20/RL_project/why-would-you-do-that/foodimg.png')
AC_img = pygame.image.load('/home/siddhss20/RL_project/why-would-you-do-that/AC.png')
TFT_img = pygame.image.load('/home/siddhss20/RL_project/why-would-you-do-that/TFT.png')
ALT_img = pygame.image.load('/home/siddhss20/RL_project/why-would-you-do-that/ALT.png')
AD_img = pygame.image.load('/home/siddhss20/RL_project/why-would-you-do-that/AD.png')
none_img = pygame.image.load('/home/siddhss20/RL_project/why-would-you-do-that/none.png')

class special_Agent():
    def __init__(self,size):
        self.clas = 'sp'
        self.id = id
        self.size = size
        self.pos = {'x':0,'y':0}
        self.food = 0
        self.age = 0
        self.strat_name = 'Intel'
        self.color = pygame.image.load('/home/siddhss20/RL_project/why-would-you-do-that/none.png')
        self.strat = basket_of_strat[self.strat_name]
        # self.reward = 0
        # self.lr = 0.3
        # self.epsilon = 0.2
        # self.gamma = 0.99
        self.state = 'Intel'
        self.hist_strat = basket_of_strat[random.choice(strat_names[0:4])]
        #self.done = False

    def next(self,pop_info,len):
        prob = [0.0,0.0,0.0,0.0,0.0]
        for key in self.pop_of_strat.keys():
            prob[strat_to_num[key]] = float(self.pop_of_strat[key][-1])/len
        return random.choices(strat_names,prob)[0]
class Agent():
    '''
        This is the class of agents. Every agent will be an instance
        of this class. Each Agent has some food at any time and we'll
        store it's age also.
    '''
    def __init__(self,size,strat):
        '''
            Initialize properties of agent
        '''
        self.clas = 'n'
        self.id = id
        self.size = size
        self.pos={'x':0,'y':0}
        self.food = 0
        self.age = 0
        self.strat_name = strat
        self.color = pygame.image.load(f'/home/siddhss20/RL_project/why-would-you-do-that/{self.strat_name}.png')
        self.strat = basket_of_strat[self.strat_name]
        self.hist_strat = self.strat

class Environment():
    '''
        This class is for the environment. An environment is modelled as
        an n by n matrix. Each cell can host 1 unit of food, or 1 agent.
        Our simulation will have a day and a night in 1 iteration. In a 
        day, our agents will be spawned at random places in the grid and
        along with some food. the goal for an agent is to aquire as much
        food as it could. At night, the agents can choose to share excess
        food with another agent or to reproduce based on some probability.
        Agents that could not aquire food in the day would die in night. 
        We'll test what strategy lead to higher chance of survival of the
        population.
    '''
    def __init__(self,food_threshold=2,n=32,foodperday=100,repChance=0.5):
        ''' 
            Initializes environment
            n : Env Matrix size
            foodperday : Food unit to spawn everyday
            repChance : prob of agents to reproduce if they have >2 unit food.
        '''
        self.n = n #matrix size
        self.foodPerDay = foodperday
        self.reproductionChance = repChance
        self.pop_hist = []
        self.agents = []
        self.grid = self.__getEmptyMat()
        self.range = range # Range in which an agent can pick food
        self.curr_total_lived=0
        self.food_pos = []
        self.food_threshold = food_threshold
        #self.sp_agentpop = 0 # population count of sp agents
        self.pop_of_strat = {'AC':[0],'TFT':[0],'ALT':[0],'AD':[0],'Intel':[0]}
        self.pop_of_size = {1:[0],2:[0]}
    def setup(self, agents : list):
        '''
            Sets environment's agents to `agents`
            agents : List of Agents initialized using `Agent` class
        '''
        self.agents=agents
        # print('population : ',len(self.agents))
        self.curr_total_lived=len(agents)
        for i in range(len(agents)):
            agents[i].id=i+1
            self.pop_of_size[agents[i].size][-1] += 1
            self.pop_of_strat[agents[i].strat_name][-1] += 1 
        self.__update()
        return
    def __update(self):
        '''
            updating the data of population.
        '''
        self.pop_of_size[1].append(self.pop_of_size[1][-1])
        self.pop_of_size[2].append(self.pop_of_size[2][-1])
        self.pop_of_strat['AC'].append(self.pop_of_strat['AC'][-1])
        self.pop_of_strat['TFT'].append(self.pop_of_strat['TFT'][-1])
        self.pop_of_strat['ALT'].append(self.pop_of_strat['ALT'][-1])
        self.pop_of_strat['AD'].append(self.pop_of_strat['AD'][-1])
        self.pop_of_strat['Intel'].append(self.pop_of_strat['Intel'][-1])
        #print(self.pop_of_strat)
        return
    def __getEmptyMat(self):
        '''
            Private Method. Initializes empty grid. 
        '''
        return np.zeros((self.n,self.n),dtype=int)
    def __populateMat(self, agent : Agent):
        '''
            Private Method. Populate grid with Agent. 
        '''
        y = agent.pos['y']
        x = agent.pos['x']
        self.grid[y][x] = agent.id
        return
    def __chooseXYrand(self):
        '''
            Private Method. Choose a cell randomly in the grid
        '''
        x,y = np.random.choice(self.n),np.random.choice(self.n)
        # If cell is occupied, choose again
        while self.grid[y][x]!=0:
            x,y = self.__chooseXYrand()
        return x,y
    def __assignPosRand(self):
        '''
            Private Method. Spawns agents randomly on the grid. 
        '''
        for i in range(len(self.agents)):
            x,y = self.__chooseXYrand()
            self.agents[i].pos = {'x':x,'y':y}
            self.__populateMat(self.agents[i])
        return
    def __populateFood(self):
        '''
            Private Method. Spawns Food randomly. 
        '''
        for i in range(self.foodPerDay):
            x,y = self.__chooseXYrand()
            self.food_pos.append((x,y))
            self.grid[y][x] = -1
        return
    def displayMat(self):
        '''
            Print current grid in numbers.
        '''
        for i in range(self.n):
            for j in range(self.n):
                print(self.grid[j][i],end=' ')
            print("")
        return
    def __pickFood(self):
        '''
           performing BFS to find nearest fox which will have food at a given cell. 
        '''
        dirn =[(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
        idlist = [agent.id for agent in self.agents]
        pos = np.argwhere(self.grid == -1).tolist()
        for inx in pos:
            que = []
            que.append(inx)
            if len(que)!=0:
                for idx in que:            #x,y = agent.pos['x'],agent.pos['y']
                    for dir in dirn:
                        i = idx[0]+dir[0]
                        j = idx[1]+dir[1]
                        chk = [i,j]             #print(type(inx))
                        if chk in que:
                            continue            #for j in range(y-r,y+r):
                        elif i>=0 and j>=0 and i<self.n and j<self.n:
                            if self.grid[i][j] != -1 and self.grid[i][j] != 0:
                                self.agents[idlist.index(self.grid[i][j])].food+=1
                                self.agents[idlist.index(self.grid[i][j])].pos['x'] =  i
                                self.agents[idlist.index(self.grid[i][j])].pos['y'] =  j
                                que.clear()
                                break
                            else:
                                que.append([i,j])

           

        return
    def choose_sp_action(self,action):
        self.spaction=action
    def reward(self):
        rew=self.pop_of_strat['Intel'][-1]-self.pop_of_strat['Intel'][-3]
        sum=0
        for st in strat_names:
            sum=sum+(self.pop_of_strat[st][-1]-self.pop_of_strat[st][-3])
        sum=sum/5
        return rew+sum
    def __night(self):
        '''
            Private Method. Checks agent's food and shares food based on
            Agent's sttrategy. Then kills the ones who couldn't precure 
            food and then creates new agents for Agents had more food.

            I am forming the probability in such a manner that big foxes are more altruistic by instinct (they already had a 1 at the beggining
            of their stratergy ) and their altruistic behaviour is further determined by what had the fox , at his mercy
            done last time he met someone and hence i am multiplying the previous stratergy of that fox also.

            conclusively the fox which has defected will have less chance of food being shared with him 
            so he will probabily die out in the long run. 
        '''
        to_delete = []
        shared_food = 0
        #a,b = 0,0
        for i in range(len(self.agents)):
            agent = self.agents[i]
            # for every agent, check it's food and strategy if it has more food
            # Increase age
            agent.age+=1
            if agent.food<1 or agent.age>10:
                to_delete.append(i)
        for i in range(len(self.agents)):
            agent = self.agents[i]
            if to_delete != []:
                if agent.food>1:
                    j = np.random.choice(to_delete)
                    atmercy = self.agents[j]
                    if agent.clas == 'sp':
                        agent.strat_name=num_to_strat[self.spaction.item()]
                        agent.hist_strat=basket_of_strat[agent.strat_name]
                        #print('j:',j ,'agent',agent.strat_name , agent.id , agent.food,'atmercy',atmercy.strat_name , atmercy.id , atmercy.food)                        
                    if agent.strat != []:    # If agent stratergy is AC or AD or ALT
                        prob = (agent.size/(atmercy.size+agent.size))*agent.strat[0]*(atmercy.hist_strat[-1] if atmercy.hist_strat != [] else 0.6)
                        agent.strat.reverse() # this line alternates the strategy of ALT type since it 
                                              #reverses the list and puts the latest used strategy at back
                                              # which others will look at  while sharing food with him.
                    else:                    # If agent is using TFT strategy
                        #if atmercy.clas == 'sp':
                            #atmercy.choose_action(agent.strat_name,self.train)
                        prob = (agent.size/(atmercy.size+agent.size))*(atmercy.hist_strat[-1] if atmercy.hist_strat != [] else 0.7)
                    choices = ['share']*(int(prob*100))+['self']*(100-int(prob*100))
                    decision = np.random.choice(choices)
                    if decision=='share':

                        atmercy.food+=1
                        agent.food-=1
                        to_delete.remove(j)
                        if agent.food == 0:
                            to_delete.append(i)

        # Kill everyone who didn't get food and have aged.
        if len(to_delete)>0:
            for i in sorted(to_delete,reverse=True):
                self.pop_of_size[self.agents[i].size][-1] -= 1
                if self.agents[i].clas == 'sp':
                    self.pop_of_strat['Intel'][-1] -= 1
                    #self.sp_agentpop -= 1
                else:
                    self.pop_of_strat[self.agents[i].strat_name][-1] -= 1

                self.agents.pop(i)
                
        # reproduce
        for agent in self.agents:
            if agent.food>=self.food_threshold:
                #*agent.strat[0] if agent.strat !=[] else 0.5
                tok = int((self.reproductionChance)*100)
                choices=['reproduce']*(tok)+['sad']*(int(100-tok))
                decision=np.random.choice(choices)
                if decision == 'reproduce':
                    agent.food -= 2
                    if agent.clas == 'sp':
                        self.agents.append(special_Agent(agent.size))
                       # self.sp_agentpop += 1
                    else:
                        self.agents.append(Agent(agent.size,agent.strat_name))
                    self.curr_total_lived+=1
                    self.agents[-1].id = self.curr_total_lived
                    self.pop_of_size[self.agents[-1].size][-1] += 1
                    self.pop_of_strat[self.agents[-1].strat_name][-1] += 1
        self.__update()

        return


    def __resetFood(self):
        '''
            Private Method. reset Agent's food numbers and matrix.
        '''
        self.grid = self.__getEmptyMat()
        self.food_pos = []
        for agent in self.agents:
            agent.food=0

    def display(self,iterate):
        '''
            Private Method. Planned to use pygame to display grid.
            animation loop would be controlled from here.
        '''
        
        x = [i for i in range(iterate+1)]
        a = self.pop_of_size[2][:-1]
        b = self.pop_of_size[1][:-1]
        s1,s2,s3,s4,s5 = self.pop_of_strat.values()
        total = [a[i] + b[i] for i in range(iterate+1)]
        #plotting begins
        plt.figure(figsize = (15,13))
        plt.subplot(2,1,1)
        plt.title('size wise population')
        plt.plot(x,a,'ro-',label = 'Agent size: 2')
        plt.plot(x,b,'bo-',label = 'Agent size: 1')
        plt.plot(x,total,'go-',label = 'total Agents')
        plt.xticks(x)
        plt.legend(loc = 'best')

        plt.subplot(2,1,2)

        plt.title('strategy wise population')
        plt.plot(x,s1[:-1],'ro-',label = 'AC')
        plt.plot(x,s2[:-1],'bo-',label = 'TFT')
        plt.plot(x,s3[:-1],'ko-',label = 'ALT')
        plt.plot(x,s4[:-1],'go-',label = 'AD')
        plt.plot(x,s5[:-1],'mo-',label = 'intelligent')
        plt.legend(loc = 'upper left')
        plt.xticks(x)
        plt.show()
        plt.close()
        return
    
    def __render(self,k):
        def population_update():
            font = pygame.font.Font('freesansbold.ttf', 15)
            AC = font.render('AC: '+str(self.pop_of_strat['AC'][-1]),True,(0,0,0))
            screen.blit(AC_img,(0,0))
            screen.blit(AC,(25,0))
            TFT = font.render('TFT: '+str(self.pop_of_strat['TFT'][-1]),True,(0,0,0))
            screen.blit(TFT_img,(0,20))
            screen.blit(TFT,(25,20))
            ALT = font.render('ALT: '+str(self.pop_of_strat['ALT'][-1]),True,(0,0,0))
            screen.blit(ALT_img,(0,40))
            screen.blit(ALT,(25,40))
            AD = font.render('AD: '+str(self.pop_of_strat['AD'][-1]),True,(0,0,0))
            screen.blit(AD_img,(0,60))
            screen.blit(AD,(25,60))
            none = font.render('Intelligent: '+str(self.pop_of_strat['Intel'][-1]),True,(0,0,0))
            screen.blit(none_img,(0,80))
            screen.blit(none,(25,80))

        pygame.init()
        pygame
        blockSize = 20 #Set the size of the grid block
        mode_size = self.n*blockSize
        screen = pygame.display.set_mode((mode_size,mode_size+100))
        pygame.display.set_caption('EVOLUTION GAME')
        ref = True
        while ref:
            screen.fill((255,255,255))
            for x in range(mode_size):
                for y in range(mode_size):
                    rect = pygame.Rect(x*blockSize, y*blockSize+100,blockSize, blockSize)
                    pygame.draw.rect(screen, gridcolor, rect, 1)
            population_update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    ref = False
            for agent in self.agents:
                #print(1)
                #print(agent.color)
                screen.blit(agent.color,(agent.pos['x']*blockSize,agent.pos['y']*blockSize+100))
            if k == 1:
                for pos in self.food_pos:
                    x,y = pos
                    screen.blit(foodimg,(x*blockSize,y*blockSize+100))
            pygame.display.update()
            pygame.time.wait(100)
            ref = False

    def iterate(self,t):
        '''
            Runs a single Iteration. Calls Internal methods in a 
            logical sequence. 
        '''
        self.__assignPosRand()
        self.__populateFood()
        if not t:
            self.__render(1)
        self.__pickFood()
        if not t:
            self.__render(2)        
        self.__night()
        if not t:
            self.__render(3)
        self.pop_hist.append(self.getPopNumber())
        # if t:
        #     self.__update_Q(self.agents)

        self.__resetFood()
        
    def getPopNumber(self):
        '''
            Print current population
        '''
        return len(self.agents)


"""# **Part B ( Q learning )**

**Training the Policy**
"""


# Training Dqn
batch_size=16
gamma=0.99
eps_st=0.9
eps_end=0.05
eps_decay=200
tgt_upd=2
cap=10000

n_ac=4
n_s=5


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


policy_net=dqn.Dqn(n_s,n_ac)
policy_net.apply(init_weights)
target_net=dqn.Dqn(n_s,n_ac)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer=optim.RMSprop(policy_net.parameters())
memory = dqn.replay_mem(cap)

stps=0
Loss=[]

def select_action(state):
    global stps
    state = state.float().unsqueeze(0)
    #print('s_a', state.dtype)
    sample = random.random()
    eps_threshold = eps_end + (eps_st - eps_end) * \
        math.exp(-1. * stps / eps_decay)
    stps += 1
    # print(sample, eps_threshold)
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_ac)]])

def optimize_model():
    if len(memory)<batch_size:
        return
    transitions=memory.sample(batch_size)
    batch=dqn.Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state if s is not None]).float()
    state_batch = torch.stack(batch.state).float()
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    #print(state_batch.shape)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(batch_size)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * gamma) + reward_batch
    criterion = nn.SmoothL1Loss()
    loss=criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    #loss=loss.clamp(max=200)
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    Loss.append(loss.detach().numpy())
    return loss


def train():
    epoch=100
    for j in range(epoch):

        # Create Env
        num_agents = 100
        foodperday = 5*num_agents
        e = Environment(foodperday=foodperday,repChance=0.8)
        print(j)

        # Initializing agents
        agents=[]
        for i in range(num_agents):
            agents.append(Agent(2,'AD'))
        for i in range(num_agents):
            agents.append(Agent(2,'AC'))
        for i in range(num_agents):
            agents.append(Agent(2,'TFT'))
        for i in range(num_agents):
            agents.append(Agent(2,'ALT'))
        for i in range(num_agents):
            agents.append(special_Agent(2))

        # pass agents in the env
        e.setup(agents)
        pop_of_strat=e.pop_of_strat
        n_eps=100
        for i in range(n_eps):
            # print(pop_of_strat['Intel'][-1])
            state=[pop_of_strat[x][-1] for x in strat_names[0:5]]
            state=[(x-(sum(state)/len(state)))/(sum(state)) for x in state]
            state=torch.tensor(state)

            action=select_action(state)
            e.choose_sp_action(action)
            e.iterate(True)
            reward=e.reward()
            #print('r',reward)
            reward = torch.LongTensor([reward])

            new_state=[pop_of_strat[x][-1] for x in strat_names[0:5]]
            new_state=[(x-(sum(new_state)/len(new_state)))/(sum(new_state)) for x in new_state]
            new_state=torch.tensor(new_state)
            memory.push(state,action,new_state,reward)
            state=new_state
            l=optimize_model()
            if(i%9==0):
                print('===>> Loss', l)
        if j%tgt_upd==0:
                target_net.load_state_dict(policy_net.state_dict())
    torch.save(policy_net.state_dict(), "model.pth")
    plt.plot([x for x in range(epoch*n_eps-15)],Loss)
    plt.show()

if __name__=="__main__":
    train()
# e.display(n_eps)

# plt.plot(np.array([Loss]),np.array([range(n_eps)]))
# plt.show()

