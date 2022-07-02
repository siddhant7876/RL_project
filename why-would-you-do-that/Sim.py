import evolgame_by_conceited_altruists as ev
import dqn
import torch
import matplotlib.pyplot as plt


"""**Testing the policy**"""
Environment =ev.Environment
Agent=ev.Agent
special_Agent=ev.special_Agent
strat_names=ev.strat_names
num_to_strat=ev.num_to_strat




policy_net=dqn.Dqn(ev.n_s,ev.n_ac,)
def sim(e2,n2,i):
    # Create Env
    # Initializing agents
    pop_of_strat = e2.pop_of_strat
    pop_of_size = e2.pop_of_size
    agents=[]
    for i in range(n2):
        agents.append(Agent(2,'AD'))
    for i in range(n2):
        agents.append(Agent(2,'AC'))
    for i in range(n2):
        agents.append(Agent(2,'TFT'))
    for i in range(n2):
        agents.append(Agent(2,'ALT'))
    for i in range(n2):
        agents.append(special_Agent(2))

    # pass agents in the env
    e2.setup(agents)
    n_eps=50
    acn=[]
    popn=[]
    for i in range(n_eps):
        state=[pop_of_strat[x][-1] for x in strat_names[0:5]]
        state=[(x-(sum(state)/len(state)))/(sum(state)) for x in state]
        state=torch.tensor(state).unsqueeze(0).float()
        print(state)
    
        v=policy_net(state)
        print('v', v)
        action=v.max(1)[1].view(1,1)
        # action=torch.tensor(random.randint(0,4))

        popn.append(pop_of_strat['Intel'][-1])
        acn.append(action.item())
        print('action= ', num_to_strat[action.item()])

        e2.choose_sp_action(action)
        e2.iterate(True)
        reward=e2.reward()
        reward = torch.tensor([reward]).float()
        #print("===> reward", reward)

        # new_state=[pop_of_strat[x][-1] for x in strat_names[0:5]]
        # new_state=[(x-(sum(new_state)/len(new_state)))/(sum(new_state)) for x in new_state]
        # new_state=torch.tensor(new_state)        
        # state=new_state

    e2.display(i+1)
    custom_palette = {}
    for q in range(n_eps):
        if acn[q] == 0:
            custom_palette[q] = 'r'
        elif acn[q] == 1:
            custom_palette[q] = 'b'
        elif acn[q] == 2:
            custom_palette[q]='k'
        else:
            custom_palette[q] = 'g'
    x=[x for x in range(n_eps)]
# import seaborn as sns
# sns.scatterplot(x,popn,data=popn,palette=custom_palette)
# plt.show()

def main(i):
    # print
    n2 = 100 # number of agents of each type
    k=5
    foodperday = n2*k
    e2 = Environment(n=35,foodperday=foodperday,repChance=0.8)
    if(i==0):
        ev.train()
    policy_net.load_state_dict(torch.load("model.pth"))
    print(policy_net.layers[-1].weight)
    policy_net.eval()
    sim(e2,n2,i)
for i in range(5):
    main(i)
# """# **Final Conclusion**

# We know that TFT and AC can dominate sometimes and AD can dominate on other ocassions . Our agent learns how to handle each agent in each environment with satisfactory accuracy. for ex , it learns that TFT is to be played against AD and AC is to be played against TFT it also learns to cooperate with other sp agents. Sometimes it may happen that our agent doesn't perform well in training but it performs very well in testing.

# # **Implementational Details**
# In the second part of the project , we have tried to teach our "special Agent" how to deal with different strategies . Our sp. agent learns this through Q learning .For analysis we first train our agents i.e obtain the optimal strategy and then test it . The Q table is initialised with (states,actions) where states are defined as whichever agent is our sp. agent interacting at the time of sharing while actions are nothing but the strategies that it considers. Here one episode is one iteration i.e one day the Q table is global that means every sp agent will refer to this table for choosing the action . By this method ,we can gain large number of experiences which is n (no. of iterations ) * p (population of sp agent).

# Rewards: our agent gets a positive reward if either the population of other agents gets low as compared to previous oteration or it's own population increases and vice-versa. thereby making the environment more competitive.

#  We also get some drawbacks in this implementation .
# Namely :

# 1.   ALT means nothing to the sp agent since it will change it's strategy in the next iteration and not get a chance to alternate .
# 2.   the current action doesn't lead to next state . That means there is no ralation between action taken and next state because the next state is only obtained in the next iteration and that is random. In other words, we cannot predict which agent is going to be 'atmercy' if we take a particular action . But we have tried to overcome this by randomising the probability of next state according to the population of the herd .Take a look at it .
# ```        
# prob = [0.0,0.0,0.0,0.0,0.0]
# for key in pop_of_strat.keys():
#         prob[strat_to_num[key]] = float(pop_of_strat[key][-1])/len
# return random.choices(strat_names,prob)[0]
# ```

# \* len is the total population while pop_of_strat gives us the info on population, strategy wise.

# So we have assumed that the next state is approximately correct and therefore we can conclude that our RL agent learns to survive accordingly.