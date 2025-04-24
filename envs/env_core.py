from collections import defaultdict
import gymnasium as gym
import numpy as np
import random
from gymnasium.spaces import Discrete, Box
from csv import writer
from pathlib import Path
# from .logistics_core import Truck, Factory, Producer
from .truck import Truck
from .factory import Factory, Producer
from .rul_gen import predictor
import datetime


class EnvCore(object):
    """
    # 环境中的智能体
    """

    def __init__(self):
        super().__init__()
        self.truck_num = 12
        self.factory_num = 50
        self.init_env()
        self.observation_space = {}
        self.action_space = {}
        self.share_observation_space = []
        share_obs_dim = 0
        self.predictor = predictor()
        obs = self._get_obs()
        for agent_id, tmp_obs in enumerate(obs):
            obs_dim = len(tmp_obs)
            self.observation_space[agent_id] = Box(low=-np.inf, high=+np.inf, shape=(obs_dim,),dtype=np.float32)
            share_obs_dim += obs_dim
            self.action_space[agent_id] = Discrete(self.factory_num+1)
        self.agent_num = self.truck_num
        self.obs_dim = len(obs[0])
        self.action_dim = self.factory_num + 1

        self.share_observation_space = [Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in range(self.agent_num)]
        self.done = []

        self.episode_num = 0
        current_date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        self.path = f"/home/lwh/Documents/Code/RL-Scheduling/result/rul/mappo/{current_date}"
        Path(self.path).mkdir(parents=True, exist_ok=True)


        self.agent_num = 12  # 设置智能体(小飞机)的个数，这里设置为两个 # set the number of agents(aircrafts), here set to two

    def reset(self, seed=None, options=None):
        '''Reset the environment state and return the initial observations for each agent.'''
        # Create folder and file to save the data
        self.make_folder()
        # init the agent
        self.init_env()
        # Count the episode num
        self.episode_num += 1
        # Get init observation
        obs = self._get_obs()
        # Episode lenth
        self.episode_len = 0
        # invalid dict
        self.invalid = []
        # done flag
        self.done = [False for _ in range(self.truck_num)]
        return obs
    
    def step(self, action_dict):
        # Set action
        self._set_action(action_dict)
        # Run the sumo until any of the agents are avaiable
        sumo_flag = True
        step_length = 0
        while sumo_flag:
            self.producer.produce_step()
            # sumo_flag = not all([tmp_truck.operable_flag for tmp_truck in self.truck_agents])
            if all([tmp_truck.operable_flag for tmp_truck in self.truck_agents]):
                sumo_flag = False
                tmp_repair_flag = [(tmp_truck.matainance_flag | tmp_truck.broken_flag)  for tmp_truck in self.truck_agents]
                if all(tmp_repair_flag):
                    sumo_flag = True

            step_length += 1
            self.episode_len += 1
            if (self.episode_len % 60) == 0:
                self.save_truck_state(self.episode_len)
        
        # Get observation, reward. record the result
        obs = self._get_obs()
        rewards = self._get_reward(action_dict)
        for tmp_invalid in self.invalid:
            rewards[tmp_invalid] = [-50]
        # Reset all indicator
        self.flag_reset()
        # Save the results
        self.save_results(self.episode_len, step_length, action_dict, rewards)
        if self.episode_len >= 7 * 24 *3600:
            self.done = [True for _ in range(self.truck_num)]
        return obs, rewards, self.done, {}
    
    def _set_action(self, action_dict):
        '''Set action for all the agent.'''
        for agent_id, action in enumerate(action_dict):
            tmp_truck = self.truck_agents[int(agent_id)]
            # Matain the truck
            if action == 50:
                tmp_truck.maintain()
            else:
                target_id = f"Factory{int(action)}"
                # Invalid action
                if target_id == tmp_truck.position:
                    self.invalid.append(agent_id)
                elif tmp_truck.state == "repair" or tmp_truck.state == "maintain":
                    pass
                else:
                    tmp_truck.delivery(destination=target_id)
    
    def _get_obs(self):
        '''Return back a dictionary for operable agents.'''
        observation = []

        fac_truck_num = defaultdict(int)
        for tmp_truck in self.truck_agents:
            fac_truck_num[tmp_truck.destination] += 1
        warehouse_storage = []
        com_truck_num = []
        for tmp_factory in self.factory.values():
            # Get the storage of material and product
            material_storage = list(tmp_factory.material_num.values())
            product_storage = tmp_factory.product_num
            warehouse_storage += material_storage + [product_storage]
            # Get the number of truck at current factory
            com_truck_num.append(fac_truck_num[tmp_factory.id])
        queue_obs = np.concatenate([warehouse_storage]+[com_truck_num])

        for tmp_truck in self.truck_agents:
            # Get the destination of all possiable route
            distance = [value for key, value in tmp_truck.map_distance.items() if key.startswith(tmp_truck.position + '_to_')]
            # Current position
            position = int(tmp_truck.position[7:])
            # Empty or not
            weight = tmp_truck.weight
            # The transported product
            product = tmp_truck.get_truck_product()
            agent_id = int(tmp_truck.id.split('_')[1])
            # Get RUL from truck's sensor reading
            rul = self.predictor.predict(tmp_truck.eng_obs)
            tmp_truck.rul = rul
            observation.append(np.concatenate([[rul]]+[queue_obs]+[distance]+[[position]]+[[weight]]+[[product]]))
        return observation

    def _get_reward(self,action_dict):
        '''Get the reward of given agents'''
        rew = []
        for agent_id in range(self.truck_num):
            tmp_truck = self.truck_agents[agent_id]
            rew.append([self._single_reward(tmp_truck)])
        return rew
    
    def _single_reward(self, agent:Truck):
        # First factor: unit profile per time step
        # agent.total_product is the record of number of total product at last time step
        rew_final_product = 0
        tmp_final_product = 0
        final_factory = ["Factory45", "Factory46", "Factory47", "Factory48", "Factory49"]
        for tmp_factory_id in final_factory:
            tmp_final_product +=  self.factory[tmp_factory_id].total_final_product
        rew_final_product = 10 * (tmp_final_product - agent.total_product)
        # Second factor: driving cost
        gk = 0.00001
        fk = 0.00002
        if agent.last_transport == 0:
            uk = gk
        else:
            uk = gk + fk
        rew_driving = uk * agent.driving_distance

        # Third factor: asset cost
        rew_ass = 0.1

        # Penalty factor
        gamma1 = 0.5
        gamma2 = 0.5
        rq = 1
        tq = agent.waiting_time
        sq = gamma1 * tq/2000 + gamma2 * (1-rq)
        if sq >= 1:
            sq = 0.99
        psq = 0.1 * np.log((1+sq)/(1-sq))

        # Short-term reward. Arrive right factory
        rew_short = 0.5 * agent.last_transport
        # Reset the short_term reward
        agent.last_transport = 0
        agent.total_product = tmp_final_product
        # Total reward
        rew = rew_final_product + rew_short - rew_driving - rew_ass - psq
        return rew
    
    def init_env(self):
        '''Generate trucks and factories'''
        map_data = np.load("envs/50f.npy",allow_pickle=True).item()
        self.truck_agents = [Truck(truck_id=f'truck_{i}', map_data=map_data) for i in range(self.truck_num)]
        # self.factory = [Factory(factory_id=f'factory_{i}', product=f'P{i}') for i in range(self.factory_num)]
        self.factory = {f'Factory{i}': Factory(factory_id=f'Factory{i}', product=f'P{i}') for i in range(self.factory_num)}

        '''Generate the final product'''
        final_product = ['A', 'B', 'C', 'D', 'E']
        remaining_materials = [f'P{i}' for i in range(45)]
        transport_idx = {}
        for i, product in enumerate(final_product):
            tmp_factory_id = f'Factory{45 + i}'
            tmp_materials = [remaining_materials.pop() for _ in range(9)]
            tmp_factory = Factory(factory_id=tmp_factory_id, product=product, material=tmp_materials)
            self.factory[tmp_factory_id] = tmp_factory
            for transport_material in tmp_materials:
                transport_idx[transport_material] = tmp_factory_id
        
        self.producer = Producer(self.factory, self.truck_agents, transport_idx)

        '''Init the factory in the begining'''
        for _ in range(100):
            self.producer.produce_step()

    def make_folder(self):
        '''Create folder to save the result'''
        # Create folder
        folder_path = self.path + '/{}/'.format(self.episode_num)
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        self.debug_files_path = folder_path + 'debug/'
        Path(self.debug_files_path).mkdir(parents=True, exist_ok=True)
        # Create file
        self.action_file = folder_path + 'action.csv'
        self.product_file = folder_path + 'product.csv'
        self.agent_file = folder_path + 'result.csv'
        self.distance_file = folder_path + 'distance.csv'
        self.truck_state = folder_path + 'truck_state.csv'

        # Create result file
        with open(self.product_file,'w') as f:
            f_csv = writer(f)
            result_list = ['time', 'step_length', 'total', 'A', 'B', 'C', 'D', 'E']
            f_csv.writerow(result_list)
        
        with open(self.agent_file, 'w') as f:
            f_csv = writer(f)
            result_list = ['time', 'step_length']
            for agent in self.truck_agents:
                agent_list = ['action_'+agent.id,'reward_'+agent.id,'cumulate reward_'+agent.id]
                result_list += agent_list
            f_csv.writerow(result_list)
        
        for truck_agent in self.truck_agents:
            debug_file = self.debug_files_path + f'{truck_agent.id}.csv'
            with open(debug_file, 'w') as f:
                f_csv = writer(f)
                result_list = ['time', 'position', 'weight', 'product','destination','driving_distance','total_distance','rul']
                f_csv.writerow(result_list)

        # Create active truck file
        with open(self.distance_file,'w') as f:
            f_csv = writer(f)
            distance_list = ['time']
            for agent in self.truck_agents:
                agent_list = [f'step_{agent.id}', f'total_{agent.id}']
                distance_list += agent_list
            f_csv.writerow(distance_list)

        with open(self.truck_state, 'w') as f:
            f_csv = writer(f)
            result_list = ['time', 'normal_num', 'broken_num', 'maintain_num', 'broken_id', 'maintain_id']
            f_csv.writerow(result_list)

    
    def flag_reset(self):
        self.invalid = []
    
    def save_results(self, time, lenth, action_dict,rewards):
        current_time = round(time/3600,3)
        with open(self.product_file, 'a') as f:
            f_csv = writer(f)
            tmp_A = round(self.factory['Factory45'].total_final_product,3)
            tmp_B = round(self.factory['Factory46'].total_final_product,3)
            tmp_C = round(self.factory['Factory47'].total_final_product,3)
            tmp_D = round(self.factory['Factory48'].total_final_product,3)
            tmp_E = round(self.factory['Factory49'].total_final_product,3)
            total = tmp_A+tmp_B+tmp_C+tmp_D+tmp_E
            product_list = [current_time,lenth,total,tmp_A,tmp_B,tmp_C,tmp_D,tmp_E]
            f_csv.writerow(product_list)

        with open(self.agent_file, 'a') as f:
            f_csv = writer(f)
            agent_list = [current_time, lenth]
            for tmp_agent in self.truck_agents:
                agent_id = int(tmp_agent.id.split('_')[1])
                if agent_id in range(self.truck_num):
                    # Write result to result file
                    tmp_agent.cumulate_reward += rewards[agent_id][0]
                    agent_list += [action_dict[agent_id], rewards[agent_id][0], tmp_agent.cumulate_reward]
                    # Write result to debug file
                    debug_file = self.debug_files_path + f'{tmp_agent.id}.csv'
                    with open(debug_file, 'a') as f:
                        f_csv_d = writer(f)
                        debug_list = [current_time, tmp_agent.position, tmp_agent.weight, tmp_agent.product, tmp_agent.destination, tmp_agent.driving_distance, tmp_agent.total_distance, tmp_agent.rul]
                        f_csv_d.writerow(debug_list)
                else:
                    agent_list += ['NA', 'NA', tmp_agent.cumulate_reward]
                
                if tmp_agent.state == 'repair' or tmp_agent.state == 'maintain':
                    debug_file = self.debug_files_path + f'{tmp_agent.id}.csv'
                    with open(debug_file, 'a') as f:
                        f_csv_d = writer(f)
                        debug_list = [current_time, tmp_agent.position, tmp_agent.weight, tmp_agent.product, tmp_agent.state, tmp_agent.driving_distance, tmp_agent.total_distance, tmp_agent.rul]
                        f_csv_d.writerow(debug_list)
            f_csv.writerow(agent_list)
        
        with open(self.distance_file, 'a') as f:
            f_csv = writer(f)
            distance_list = [current_time]
            for tmp_agent in self.truck_agents:
                distance_list += [tmp_agent.driving_distance, tmp_agent.total_distance]
            f_csv.writerow(distance_list)
        
        self.save_truck_state(time)
        
    def save_truck_state(self, time):
        current_time = round(time/3600,3)
        with open(self.truck_state, 'a') as f:
            f_csv = writer(f)
            truck_state_list = [current_time]
            normal_num = 0
            broken_num = 0
            maintain_num = 0
            broken_id = []
            maintain_id = []
            for tmp_agent in self.truck_agents:
                if tmp_agent.state == 'repair':
                    broken_num += 1
                    broken_id.append(tmp_agent.id)
                elif tmp_agent.state == 'maintain':
                    maintain_num += 1
                    maintain_id.append(tmp_agent.id)
            normal_num = self.truck_num - broken_num - maintain_num
            truck_state_list += [normal_num, broken_num, maintain_num]
            truck_state_list += [broken_id, maintain_id]
            f_csv.writerow(truck_state_list)