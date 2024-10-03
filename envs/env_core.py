from collections import defaultdict
import gymnasium as gym
import numpy as np
import random
from gymnasium.spaces import Discrete, Box
from csv import writer
from pathlib import Path
from .logistics_core import Truck, Factory, Producer
import string
import xml.etree.ElementTree as ET

class EnvCore(object):
    """
    # 环境中的智能体
    """

    def __init__(self):
        super().__init__()
        self.truck_num = 30
        self.factory_num = 50
        self.init_env()
        self.observation_space = {}
        self.action_space = {}
        self.share_observation_space = []
        share_obs_dim = 0
        obs = self._get_obs()
        for agent_id, tmp_obs in obs.items():
            obs_dim = len(tmp_obs)
            self.observation_space[agent_id] = Box(low=0, high=+np.inf, shape=(obs_dim,),dtype=np.float32)
            share_obs_dim += obs_dim
            self.action_space[agent_id] = Discrete(self.factory_num)
        self.agent_num = self.truck_num
        self.obs_dim = len(obs[0])
        self.action_dim = self.factory_num

        self.share_observation_space = [Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in range(self.agent_num)]
        self.done = {}

        self.episode_num = 0
        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        self.path = f"/home/lwh/Documents/Code/RL-Scheduling/result/async_mappo/exp_{random_string}"

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
        self.done['__all__'] = False
        return obs, {}
    
    def step(self, action_dict):
        # Set action
        self._set_action(action_dict)
        # Run SUMO until all the agents are avaiable
        sumo_flag = True
        # Record step lenth
        step_lenth = 0
        # Simulation
        while sumo_flag:
            [tmp_truck.refresh_state() for tmp_truck in self.truck_agents]
            self.producer.produce_all()
            self.producer.unload_all()
            self.producer.load_all()
            trucks_operable = [tmp_truck.operable_flag for tmp_truck in self.truck_agents]
            # If any of the trucks are operable, break the loop
            sumo_flag = False if any(trucks_operable) else True
            step_lenth += 1
            self.episode_len += 1

        # Get observation, reward. record the result
        obs = self._get_obs()
        rewards = self._get_reward(action_dict)
        rewards.update({tmp_key:-50 for tmp_key in self.invalid})
        # Reset all indicator
        self.flag_reset()
        # Save the results
        self.save_results(self.episode_len, step_lenth, action_dict, rewards)
        if self.episode_len >= 7 * 24 *3600:
            self.done['__all__'] = True
        return obs, rewards, self.done, self.done, {}
    
    def _set_action(self, action_dict:dict):
        '''Set action for all the agent.'''
        for agent_id, action in action_dict.items():
            agent = self.truck_agents[int(agent_id)]
            target_id = self.factory[int(action)].id
            # Invalid action
            if target_id == agent.position:
                self.invalid.append(agent_id)
            else:
                # Assign truck to the new destination
                agent.delivery(destination=target_id)
    
    def _get_obs(self):
        '''Return back a dictionary for operable agents.'''
        observation = {}

        fac_truck_num = defaultdict(int)
        for tmp_truck in self.truck_agents:
            fac_truck_num[tmp_truck.destination] += 1
        warehouse_storage = []
        com_truck_num = []
        for tmp_factory in self.factory:
            # Get the storage
            tmp_storage = tmp_factory.warehouse['Quantity'].tolist()
            warehouse_storage += tmp_storage
            # Get the number of truck at current factory
            com_truck_num.append(fac_truck_num[tmp_factory.id])
        queue_obs = np.concatenate([warehouse_storage]+[com_truck_num])

        # Generate observation for those operable trucks
        operable_trucks = [tmp_truck for tmp_truck in self.truck_agents if tmp_truck.operable_flag]
        for tmp_truck in operable_trucks:
            # Get the destination of all possiable route
            distance = [value for key, value in tmp_truck.map_distance.items() if key.startswith(tmp_truck.position + '_to_')]
            # Current position
            position = int(tmp_truck.position[7:])
            # Empty or not
            weight = tmp_truck.weight
            # The transported product
            product = tmp_truck.get_truck_produce()
            agent_id = int(tmp_truck.id.split('_')[1])
            observation[agent_id] = np.concatenate([[position]]+[queue_obs]+[distance]+[[weight]]+[[product]])
        return observation

    def _get_reward(self,action_dict:dict):
        '''Get the reward of given agents'''
        rew = {}
        for agent_id in action_dict.keys():
            tmp_truck = self.truck_agents[agent_id]
            rew[agent_id] = self._single_reward(tmp_truck)
        return rew
    
    def _single_reward(self, agent:Truck):
        # First factor: unit profile
        rew_final_product = 0
        tmp_final_product = 0
        for tmp_factory in self.factory:
            tmp_final_product +=  tmp_factory.total_final_product
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
        tq = agent.time_step
        sq = gamma1 * tq/2000 + gamma2 * (1-rq)
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
        '''Generate Truck and factory'''
        # Get the lane id
        # parking_dict = self.xml_dict("map/sg_map/factory.prk.xml")
        # Get map data
        map_data = np.load("envs/50f.npy",allow_pickle=True).item()
        # Truck agents
        self.truck_agents = [Truck(truck_id='truck_'+str(i), map_data=map_data) for i in range(self.truck_num)]
        # Add factory 0~44
        self.factory = [Factory(factory_id=f'Factory{i}',material=[f'P{i}'],product={f'P{i}':{}}) for i in range(45)]
        # Generate raw material and product list
        final_products = ['A', 'B', 'C', 'D', 'E']
        remaining_materials = [f'P{i}' for i in range(45)]
        transport_idx = {}
        for i, product in enumerate(final_products):
            tmp_factory_id = f'Factory{45 + i}'
            tmp_materials = [remaining_materials.pop() for _ in range(9)]
            product_requirement = {product:{mat:1 for mat in tmp_materials}}
            tmp_factory = Factory(factory_id=tmp_factory_id,material=tmp_materials,product=product_requirement)
            self.factory.append(tmp_factory)
            for transport_material in tmp_materials:
                transport_idx[transport_material] = tmp_factory_id
        
        self.producer = Producer(self.factory, self.truck_agents,transport_idx)
        for _ in range(100):
            self.producer.produce_all()

    def make_folder(self):
        '''Create folder to save the result'''
        # Create folder
        folder_path = self.path + '/{}/'.format(self.episode_num)
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        # Create file
        self.product_file = folder_path + 'product.csv'
        self.agent_file = folder_path + 'result.csv'
        self.distance_file = folder_path + 'distance.csv'
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

        # Create active truck file
        with open(self.distance_file,'w') as f:
            f_csv = writer(f)
            distance_list = ['time']
            for agent in self.truck_agents:
                agent_list = [f'step_{agent.id}', f'total_{agent.id}']
                distance_list += agent_list
            f_csv.writerow(distance_list)

    def xml_dict(self, xml_file) -> dict:
        '''
        Get data from xml file.
        Use in the traci api
        '''
        tree = ET.parse(xml_file)
        root = tree.getroot()
        parking_dict = {}
        for parking_area in root.findall('parkingArea'):
            parking_id = parking_area.get('id')
            lane = parking_area.get('lane')
            if lane.endswith('_0'):
                lane = lane[:-2]
            parking_dict[parking_id] = lane
        return parking_dict
    
    def flag_reset(self):
        self.invalid = []
    
    def save_results(self, time, lenth, action_dict,rewards):
        current_time = round(time/3600,3)
        with open(self.product_file, 'a') as f:
            f_csv = writer(f)
            tmp_A = round(self.factory[45].warehouse.loc['A','Quantity'],3)
            tmp_B = round(self.factory[46].warehouse.loc['B','Quantity'],3)
            tmp_C = round(self.factory[47].warehouse.loc['C','Quantity'],3)
            tmp_D = round(self.factory[48].warehouse.loc['D','Quantity'],3)
            tmp_E = round(self.factory[49].warehouse.loc['E','Quantity'],3)
            total = tmp_A+tmp_B+tmp_C+tmp_D+tmp_E
            product_list = [current_time,lenth,total,tmp_A,tmp_B,tmp_C,tmp_D,tmp_E]
            f_csv.writerow(product_list)

        with open(self.agent_file, 'a') as f:
            f_csv = writer(f)
            agent_list = [current_time, lenth]
            for tmp_agent in self.truck_agents:
                agent_id = int(tmp_agent.id.split('_')[1])
                if agent_id in action_dict.keys():
                    tmp_agent.cumulate_reward += rewards[agent_id]
                    agent_list += [action_dict[agent_id], rewards[agent_id], tmp_agent.cumulate_reward]
                else:
                    agent_list += ['NA', 'NA', tmp_agent.cumulate_reward]
            f_csv.writerow(agent_list)
        
        with open(self.distance_file, 'a') as f:
            f_csv = writer(f)
            distance_list = [current_time]
            for tmp_agent in self.truck_agents:
                distance_list += [tmp_agent.driving_distance, tmp_agent.total_distance]
            f_csv.writerow(distance_list)