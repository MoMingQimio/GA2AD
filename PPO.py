import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical


################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.action_dim = action_dim
        if has_continuous_action_space:

            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 128),
                            nn.Tanh(),
                            nn.Linear(128, 128),
                            nn.Tanh(),
                            nn.Linear(128, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 128),
                            nn.Tanh(),
                            nn.Linear(128, 128),
                            nn.Tanh(),
                            nn.Linear(128, action_dim[0]+action_dim[1]),
                            nn.Softmax(dim=-1)
                        )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 128),
                        nn.Tanh(),
                        nn.Linear(128, 128),
                        nn.Tanh(),
                        nn.Linear(128, 1)
                    )
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):

        # if self.has_continuous_action_space:
        #     action_mean = self.actor(state)
        #     cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        #     dist = MultivariateNormal(action_mean, cov_mat)
        # else:
        action_probs = self.actor(state)
        
        if torch.isnan(action_probs).any():
            action_probs = torch.rand_like(action_probs)
        # dist = Categorical(action_probs)
        #print(action_probs)
        action_probs1 = action_probs[:,:self.action_dim[0]]
        action_probs2 = action_probs[:,self.action_dim[0]:]
        dist1 = Categorical(action_probs1)
        dist2 = Categorical(action_probs2)


        action1 = dist1.sample()
        action2 = dist2.sample()
        action_logprob1 = dist1.log_prob(action1)
        action_logprob2 = dist2.log_prob(action2)
        action = torch.concat([action1, action2])
        action_logprob = torch.concat([action_logprob1, action_logprob2])
        # action = dist.sample()
        # action_logprob = dist.log_prob(action)
        state_val = self.critic(state)



        # return action1.detach(),action2.detach(),  action_logprob1.detach(),action_logprob2.detach(), state_val.detach()

        return action.detach(),  action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):

        # if self.has_continuous_action_space:
        #     action_mean = self.actor(state)
        #
        #     action_var = self.action_var.expand_as(action_mean)
        #     cov_mat = torch.diag_embed(action_var).to(device)
        #     dist = MultivariateNormal(action_mean, cov_mat)
        #
        #     # For Single Action Environments.
        #     if self.action_dim == 1:
        #         action = action.reshape(-1, self.action_dim)
        # else:

        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action_probs = self.actor(state)
        #如果action_probs 中存在NAN
        if torch.isnan(action_probs).any():
            action_probs = torch.rand_like(action_probs)
        action_probs1 = action_probs[:,:, :self.action_dim[0]]
        action_probs2 = action_probs[:,:, self.action_dim[0]:]
        dist1 = Categorical(action_probs1)
        dist2 = Categorical(action_probs2)

            #dist = Categorical(action_probs)

        # action_logprobs = dist.log_prob(action)
        action_logprobs1 = dist1.log_prob(action[:,0])
        action_logprobs2 = dist2.log_prob(action[:,1])
        #action_logprobs1 = action_logprobs1.unsqueeze(2)
        # action_logprobs = torch.concat([action_logprobs1.unsqueeze(2),action_logprobs2.unsqueeze(2)],dim=-1)
        action_logprobs = torch.concat([action_logprobs1, action_logprobs2]).reshape(action_logprobs1.shape[1],2)


        # dist_entropy = dist.entropy()
        dist_entropy1 = dist1.entropy()
        dist_entropy2 = dist2.entropy()
        dist_entropy = dist_entropy1+dist_entropy2
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,  action_std_init=0.6):



        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                # state = torch.FloatTensor(state).to(device)
                # state = state.to(device)
                state = torch.Tensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)
                #action1, action2, action_logprob1, action_logprob_2, state_val = self.policy_old.act(state)
                # action = [action1, action2]
                # action_logprob = [action_logprob1, action_logprob_2]
            # self.buffer.states.append(state)
            # self.buffer.actions.append(action)
            # self.buffer.logprobs.append(action_logprob)
            # self.buffer.state_values.append(state_val)

            return action, action_logprob, state_val

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages.unsqueeze(1)
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages.unsqueeze(1)

            # final loss of clipped objective PPO
            loss = torch.mm(-torch.min(surr1, surr2),torch.tensor([[0.5],[0.5]]).to(device)).squeeze()+ 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy.squeeze()
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       


