import gymnasium as gym
import traci
import numpy as np
from gymnasium.spaces import Discrete, Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.env_context import EnvContext
from csv import writer
from pathlib import Path
from .core import Truck, Factory, product_management

class Simple_Scheduling(MultiAgentEnv):
    def __init__(self, env_config:EnvContext):
        # 12 Trucks, 4 Factories. The last factory is not agent
        self.truck_num = env_config['agents']
        self.factory_num = 4
        # init sumo at the begining
        self.init_sumo()
        # Define the observation space and action space.
        self.observation_space = {}
        self.action_space = {}
        obs = self._get_obs()
        for agent_id, tmp_obs in obs.items():
            obs_dim = len(tmp_obs)
            self.observation_space[agent_id] = Box(low=0, high=+np.inf, shape=(obs_dim,),dtype=np.float32)
            self.action_space[agent_id] = Discrete(4)
        # The done flag
        self.done = {}

        self.episode_num = 0
        self.path = env_config['path'] + f'/{env_config.worker_index}'

    def reset(self):
        '''
        Reset the environment state and return the initial observations for each agent.
        '''
        # Create folder and file to save the data
        self.make_folder()
        # Count the episode num
        self.episode_num += 1
        # Init SUMO
        self.init_sumo()
        # Get init observation
        obs = self._get_obs()
        # Operable penalty
        self.operable_penalty = {}
        # done flag
        self.done['__all__'] = False
        return obs

    def step(self, action_dict):
        '''
        Compute the environment dynamics given the actions of each agent.
        Return a dictionary of observations, rewards, dones (indicating whether the episode is finished), and info.
        '''
        # Reset the penalty before excute action
        self.operable_penalty = {}
        # Set action
        self._set_action(action_dict)
        # Run SUMO until all the agents are avaiable
        sumo_flag = True
        # Record step lenth
        step_lenth = 0
        # The SUMO simulation
        while sumo_flag:
            traci.simulationStep()
            # Refresh truck state
            tmp_state = [tmp_truck.refresh_state() for tmp_truck in self.truck_agents]
            self.manager.rl_produce_load()
            trucks_operable = [tmp_truck.operable_flag for tmp_truck in self.truck_agents]
            # If all the trucks are operable, break the loop
            sumo_flag = False if all(trucks_operable) else True
            step_lenth += 1

        obs = self._get_obs()
        rewards = self._get_reward(action_dict.keys())
        # Reset the flag
        self.flag_reset()

        # Save the results
        current_time = traci.simulation.getTime()

        with open(self.result_file, 'a') as f:
            f_csv = writer(f)
            tmp_A = round(self.factory[2].product.loc['A','total'],3)
            tmp_B = round(self.factory[3].product.loc['B','total'],3)
            tmp_P12 = round(self.factory[1].product.loc['P12','total'],3)
            tmp_P23 = round(self.factory[2].product.loc['P23','total'],3)
            tmp_time = round(current_time / 3600,3)
            result_list = [tmp_time,step_lenth,tmp_A,tmp_B,tmp_P12,tmp_P23]
            for action, agent, reward in zip(action_dict.values(), self.truck_agents, rewards.values()):
                agent.cumulate_reward += reward
                if type(action) is np.int32:
                    act_int = action
                else:
                    act_int = np.argmax(action)
                reward_list = [act_int, reward, agent.cumulate_reward]
                result_list += reward_list
            f_csv.writerow(result_list)
            
        with open(self.active_truck_file, 'a') as f:
            f_csv = writer(f)
            total_num = 0
            truck_state = []
            for tmp_truck in self.truck_agents:
                truck_state += [tmp_truck.state,tmp_truck.operable_flag]
                if tmp_truck.state != "waitting":
                    total_num += 1
            tmp_time = round(current_time / 3600,3)
            act_list = [tmp_time, total_num]
            act_list += truck_state
            f_csv.writerow(act_list)

        if current_time >= 3600*24:
            self.done['__all__'] = True

        return obs, rewards, self.done, {}

    def _get_obs(self) -> dict:
        '''
        Return back a dictionary for operable agents
        '''
        observation = {}
        # Shared observation, Storage/Queue from factory
        product_storage = []
        material_storage = []
        com_truck_num = []
        for factory_agent in self.factory:
            # Get the storage of product
            for factory_product in factory_agent.product.index:
                product_storage.append(factory_agent.container.loc[factory_product,'storage'])
            # Get the storage of the material
            material_index = factory_agent.get_material()
            for tmp_material in material_index:
                material_storage.append(factory_agent.container.loc[tmp_material,'storage'])
            # Get the number of trucks at current factories
            tmp_truck_num = 0
            for tmp_truck in self.truck_agents:
                if tmp_truck.destination == factory_agent.id:
                    tmp_truck_num += 1
            com_truck_num.append(tmp_truck_num)
        queue_obs = np.concatenate([product_storage] +[material_storage] + [com_truck_num])

        # The truck agents' observation
        for truck_agent, agent_id in zip(self.truck_agents,range(len(self.truck_agents))):
            # axis of the agent
            axis = truck_agent.get_axis()
            # Current destination
            destination = int(truck_agent.destination[-1])
            # The state of the truck
            state = truck_agent.get_truck_state()
            # The transported product
            product = truck_agent.get_truck_produce()

            observation[agent_id] = np.concatenate([queue_obs] + [axis] + [[destination]] + [[product]] + [[state]])
        
        return observation
    
    def _set_action(self, actions:dict):
        '''
        Set action for all the agent
        '''
        for agent_id, action in actions.items():
            agent = self.truck_agents[agent_id]
            if type(action) is np.int32:
                act_int = action
            else:
                act_int = np.argmax(action)
            target_id = self.factory[act_int].id
            # Assign truck to the new destination
            if agent.operable_flag:
                agent.delivery(destination=target_id)
            else:
                pass
    
    def _get_reward(self, act_keys) -> dict:
        '''
        Get reward for given agents
        '''
        rew = {}
        for agent_id in act_keys:
            tmp_agent = self.truck_agents[agent_id]
            rew[agent_id] = self.truck_reward(tmp_agent)
        return rew

    def truck_reward(self, agent) -> float:
        '''
        Calculate reward for the given truck agent.
        The reward depends on the waitting time and the number of product transported during last time step
        '''

        # Reward 1: final product * 40
        rew_final_product = 0
        for factory in self.factory:
            rew_final_product += 4 * factory.step_final_product
        # Reward 2: Transported component durning last time step
        rew_last_components = 0.1 * agent.last_transport
        
        # Reward 3: depends on the distance of between trucks and the destination 0~8
        distance = agent.get_distance(agent.destination)
        if distance < 0:
            distance = 1
        # Normalize the distance (min-max scale), assume maximum distance is 5000
        norm_distance = distance / 5000
        distance_reward = -3 * np.log(norm_distance)

        '''
        # Penalty, when the truck is idle['time','total number','running turck']
        if agent.state == "waitting":
            penalty = -20
        '''

        rew = rew_final_product + rew_last_components
        # print("rew: {} ,rew_1: {} ,rew_2: {} ,penalty: {} ,long_rew: {}".format(rew,rew_1,rew_2,penalty,long_rew))
        return rew
    
    def shared_reward(self) -> float:
        '''
        Long-term shared reward
        '''
        rew_trans = 0
        rew_product = 0

        # Reward 1: Total transportated product during last time step, 0-50
        # Reward 2: Number of final product 0-50
        # P1=1, P2=10
        for factory in self.factory:
            rew_trans +=  1 * factory.step_transport
            rew_product += 40 * factory.step_final_product
        shared_rew = rew_trans + rew_product
        return shared_rew

    def init_sumo(self):
        try:
            traci.close()
            print('restart sumo')
        except:
            pass
        traci.start(["sumo", "-c", "map/3km_1week/osm.sumocfg","--threads","20","--no-warnings","True"])
        self.truck_agents = [Truck(truck_id='truck_'+str(i)) for i in range(self.truck_num)]
        self.factory = [Factory(factory_id='Factory0', produce_rate=[['P1',5,None,None]]),
                Factory(factory_id='Factory1', produce_rate=[['P2',10,None,None],['P12',2.5,'P1,P2','1,1']]),
                Factory(factory_id='Factory2', produce_rate=[['P3',5,None,None],['P23',2.5,'P2,P3','1,1'],['A',2.5,'P12,P3','1,1']]),
                Factory(factory_id='Factory3', produce_rate=[['P4',5,None,None],['B',2.5,'P23,P4','1,1']])]
        self.manager = product_management(self.factory, self.truck_agents)
        for _ in range(100):
            traci.simulationStep()
            tmp_state = [tmp_truck.refresh_state() for tmp_truck in self.truck_agents]
            # self.manager.produce_load()

    def make_folder(self):
        '''
        Create folder to save the result
        '''
        # Create folder
        folder_path = self.path + '/{}/'.format(self.episode_num)
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        # Create file
        self.result_file = folder_path + 'result.csv'
        self.active_truck_file = folder_path + 'active_truck.csv'
        # Create result fileflag_reset
        with open(self.result_file,'w') as f:
            f_csv = writer(f)
            result_list = ['time','step_lenth','A','B','P12','P23']
            for agent in self.truck_agents:
                agent_list = ['action_'+agent.id,'reward_'+agent.id,'cumulate reward_'+agent.id]
                result_list += agent_list
            f_csv.writerow(result_list)
        # Create active truck file
        with open(self.active_truck_file,'w') as f:
            f_csv = writer(f)
            act_truck_list = ['time','total number']
            for agent in self.truck_agents:
                agent_list = [f'state_{agent.id}',f'operable_{agent.id}']
                act_truck_list += agent_list
            f_csv.writerow(act_truck_list)

    def flag_reset(self):
        for factory_agent in self.factory:
            # The number of pruduced component during last time step
            factory_agent.step_produced_num = 0
            factory_agent.step_final_product = 0
            # The number of decreased component during last time step
            factory_agent.step_transport = 0
            # factory_agent.step_emergency_product = {'Factory0':0, 'Factory1':0, 'Factory2':0, 'Factory3':0}
        for truck in self.truck_agents:
            # Reset the number of transported goods
            truck.last_transport = 0

    def stop_env(self):
        traci.close()
