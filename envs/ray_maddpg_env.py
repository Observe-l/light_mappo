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
        self.truck_num = 12
        self.factory_num = 3
        # init sumo at the begining
        self.init_sumo()
        # Define the observation space and action space.
        self.observation_space = {}
        self.action_space = {}
        obs = self._get_obs()
        for agent_id, tmp_obs in obs.items():
            obs_dim = len(tmp_obs)
            self.observation_space[agent_id] = Box(low=0, high=+np.inf, shape=(obs_dim,),dtype=np.float32)
            if agent_id < self.truck_num:
                self.action_space[agent_id] = Discrete(3)
            else:
                self.action_space[agent_id] = Discrete(2)
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
        # done flag
        self.done['__all__'] = False

        return obs

    def step(self, action_dict):
        '''
        Compute the environment dynamics given the actions of each agent.
        Return a dictionary of observations, rewards, dones (indicating whether the episode is finished), and info.
        '''
        # Set action
        self._set_action(action_dict)
        # The SUMO simulation
        for _ in range(500):
            traci.simulationStep()
            # Refresh truck state
            tmp_state = [tmp_truck.refresh_state() for tmp_truck in self.truck_agents]
            self.manager.produce_load()
        
        # Resume all trucks to get observation
        self.resume_truck()
        obs = self._get_obs()
        rewards = self._get_reward()
        # Park all truck to continue the simulation
        self.park_truck()
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
            f_csv.writerow([tmp_time,tmp_A,tmp_B,tmp_P12,tmp_P23])
        for agent, reward, tmp_file in zip(self.truck_agents + self.factory_agents, rewards.values(), self.reward_file):
            agent.cumulate_reward += reward
            with open(tmp_file,'a') as f:
                f_csv = writer(f)
                tmp_time = round(current_time / 3600,3)
                f_csv.writerow([tmp_time, reward, agent.cumulate_reward])

        if current_time >= 3600*24:
            self.done['__all__'] = True
        return obs, rewards, self.done, {}

    def _get_obs(self) -> dict:
        '''
        Return back a dictionary for both truck and factory agents
        '''
        observation = {}
        # The agent id. must be integer, start from 0
        agent_id = 0
        # The truck agents' observation
        for truck_agent in self.truck_agents:
            distance = []
            com_truck_num = []
            com_factory_action = []

            for factory_agent in self.factory_agents:
                # Observation 1: distance to 3 factories, [0,+inf]
                tmp_distance = truck_agent.get_distance(factory_agent.id)
                if tmp_distance < 0:
                    tmp_distance = 0
                distance.append(tmp_distance)

                # Observation 3: the action of factory agent
                tmp_factory_action = 1 if factory_agent.req_truck is True else 0
                com_factory_action.append(tmp_factory_action)

            # Observation 2: number of trucks that driving to each factory
            for factory in self.factory:
                tmp_truck_num = 0
                for tmp_truck in self.truck_agents:
                    if tmp_truck.destination == factory.id:
                        tmp_truck_num += 1
                com_truck_num.append(tmp_truck_num)

            # Observation 4: The state of the truck
            state = truck_agent.get_truck_state()
            # Store the observation in the dictionary
            observation[agent_id] = np.concatenate([distance] + [com_truck_num] + [com_factory_action] + [[state]])

            agent_id += 1
        
        # The factory agents' observation
        for factory_agent in self.factory_agents:
            # Get the storage of product
            product_storage = []
            for factory_product in factory_agent.product.index:
                product_storage.append(factory_agent.container.loc[factory_product,'storage'])
            # Get the storage of the material
            material_storage = []
            material_index = factory_agent.get_material()
            for tmp_material in material_index:
                material_storage.append(factory_agent.container.loc[tmp_material,'storage'])
            # Get the number of trucks at current factories
            truck_num = 0
            for tmp_truck in self.truck_agents:
                if tmp_truck.destination == factory_agent.id:
                    truck_num += 1
            observation[agent_id] = np.concatenate([product_storage] + [material_storage] + [[truck_num]])

            agent_id += 1
        
        return observation
    
    def _set_action(self, actions:dict):
        '''
        Set action for all the agent
        '''
        # print("action is:", actions)
        for agent_id, action in actions.items():
            if agent_id < self.truck_num:
                agent = self.truck_agents[agent_id]
                target_id = self.factory_agents[np.argmax(action)].id
                if agent.operable_flag:
                    agent.delivery(destination=target_id)
                else:
                    pass
            else:
                self.factory_agents[agent_id-self.truck_num].req_truck = True if np.argmax(action)==0 else False
    
    def _get_reward(self) -> dict:
        '''
        Get reward for all agents
        '''
        rew = {}
        # The agent id. must be integer, start from 0
        agent_id = 0
        for tmp_agent in self.truck_agents:
            rew[agent_id] = self.truck_reward(tmp_agent)
            agent_id += 1
        for tmp_agent in self.factory_agents:
            rew[agent_id] = self.factory_reward(tmp_agent)
            agent_id += 1
        return rew

    def truck_reward(self, agent) -> float:
        '''
        Calculate reward for the given truck agent.
        The reward depends on the waitting time and the number of product transported during last time step
        '''

        # Short-term reward 1: change of transported product in next factory 0~100
        rew_1 = 0
        penalty = 0
        for factory in self.factory:
            rew_1 += factory.step_emergency_product[agent.destination]

            # Penalty: going to wrong factory
            if agent.destination == factory.id and factory.req_truck is False:
                penalty = -100

        
        # Short-term reward 2: depends on the distance of between trucks and the destination 0~8
        distance = agent.get_distance(agent.destination)
        if distance < 0:
            distance = 1
        # Normalize the distance (min-max scale), assume maximum distance is 5000
        norm_distance = distance / 5000
        rew_2 = -3 * np.log(norm_distance)

        # Get shared Long-term reward
        long_rew = self.shared_reward()
        
        rew = rew_1 + rew_2 + penalty  + long_rew
        # print("rew: {} ,rew_1: {} ,rew_2: {} ,penalty: {} ,long_rew: {}".format(rew,rew_1,rew_2,penalty,long_rew))
        return rew
    
    def factory_reward(self, agent) -> float:
        '''
        Read the reward from factory agent.
        '''
        # Short-term reward 1: change of production num 0~20
        rew_1 = 1 * agent.step_transport

        # Short-term reward 2: change of transported product in next factory, 0~100
        # Penalty: when the factory run out of material, 0~50
        rew_2 = 0
        penalty_1 = 0
        for factory in self.factory:
            rew_2 += factory.step_emergency_product[agent.id]
            penalty_1 += 2 * factory.penalty[agent.id]
        
        # penalty: if more than half trucks in same factory, and factory still ask for new truck
        penalty_2 = 0
        tmp_truck_num = 0
        for truck_agent in self.truck_agents:
            if truck_agent.destination == agent.id:
                tmp_truck_num += 1
        
        if tmp_truck_num >= 0.5 * len(self.truck_agents) and agent.req_truck:
            penalty_2 = -200


        # Get shared Long-term reward
        long_rew = self.shared_reward()

        rew = rew_1 + rew_2 + long_rew + penalty_1 + penalty_2
        # print("rew: {} ,rew_1: {} ,rew_2: {} ,penalty_1: {} ,penalty_2: {} ,long_rew: {}".format(rew,rew_1,rew_2,penalty_1,penalty_2,long_rew))
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
            rew_product += 4 * factory.step_final_product
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
        self.factory_agents = self.factory[0:self.factory_num]
        self.manager = product_management(self.factory, self.truck_agents)
        for _ in range(100):
            traci.simulationStep()
            tmp_state = [tmp_truck.refresh_state() for tmp_truck in self.truck_agents]
            self.manager.produce_load()

    def make_folder(self):
        '''
        Create folder to save the result
        '''
        # Create folder
        folder_path = self.path + '/{}/'.format(self.episode_num)
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        # Create file
        self.result_file = folder_path + 'result.csv'
        self.reward_file = []
        # Create reward file
        for agent in self.truck_agents + self.factory_agents:
            tmp_path = folder_path + agent.id + '.csv'
            with open(tmp_path,'w') as f:
                f_csv = writer(f)
                f_csv.writerow(['time','reward','cumulate reward'])
            self.reward_file.append(tmp_path)
        # Create result file
        with open(self.result_file,'w') as f:
            f_csv = writer(f)
            f_csv.writerow(['time','A','B','P12','P23'])
    

    def resume_truck(self):
        '''
        resume all truck from parking area to get the distance
        '''
        for agent in self.truck_agents:
            tmp_pk = traci.vehicle.getStops(vehID=agent.id)
            if len(tmp_pk) > 0:
                latest_pk = tmp_pk[0]
                if latest_pk.arrival > 0:
                    traci.vehicle.resume(vehID=agent.id)
        traci.simulationStep()
    
    def park_truck(self):
        '''
        put all truck back to the parking area
        '''
        for agent in self.truck_agents:
            try:
                traci.vehicle.setParkingAreaStop(vehID=agent.id, stopID=agent.destination)
            except:
                pass
        for _ in range(5):
            traci.simulationStep()
    
    def flag_reset(self):
        for factory_agent in self.factory:
            # The number of pruduced component during last time step
            factory_agent.step_produced_num = 0
            factory_agent.step_final_product = 0
            # The number of decreased component during last time step
            factory_agent.step_transport = 0
            factory_agent.step_emergency_product = {'Factory0':0, 'Factory1':0, 'Factory2':0, 'Factory3':0}
            # The penalty, when run out of material
            factory_agent.penalty = {'Factory0':0, 'Factory1':0, 'Factory2':0, 'Factory3':0}
    def stop_env(self):
        traci.close()
