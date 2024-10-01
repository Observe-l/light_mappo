import numpy as np
import pandas as pd
import random

class Truck(object):
    '''
    The class of the truck
    '''
    def __init__(self,truck_id:str = 'truck_0', capacity:float = 5.0, weight:float = 0.0,\
                 state:str = 'delivery', product:str = 'P1', map_data:dict = None) -> None:
        self.id = truck_id
        self.capacity = capacity
        self.map_time = map_data['time']
        self.map_distance = map_data['distance']
        self.route = random.choice(list(self.map_distance.keys()))
        self.position, self.destination = self.route.split('_to_')
        self.reset(weight, state, product)
    
    def reset(self, weight:float = 0.0, state:str = 'delivery', product:str = 'A') -> None:
        self.weight = weight
        self.state = state
        self.product = product

        self.operable_flag = True

        # record total transported product
        self.total_product = 0.0
        self.last_transport = 0.0
        # count time step
        self.time_step = 0
        # Record the reward
        self.cumulate_reward = 0.0
        # reset the driving distance
        self.driving_distance = 0.0
        self.total_distance = 0.0
    
    def refresh_state(self) -> None:
        self.time_step += 1
        if self.time_step >= self.map_time[self.route]:
            self.driving_distance = self.map_distance[self.route]
            self.total_distance += self.driving_distance
            self.position = self.destination
            if self.weight == 0:
                self.state = 'waiting'
                self.operable_flag = True
            else:
                self.state = 'arrived'
                self.operable_flag = True

    
    def delivery(self, destination:str) -> None:
        '''
        Select a route, change the state, reset time
        '''
        self.destination = destination
        self.route = f'{self.position}_to_{self.destination}'
        self.time_step = 0
        self.state = 'delivery'
        self.operable_flag = False

    def load_cargo(self, product:str) -> None:
        '''
        Load cargo to the truck. Cannot exceed the maximum capacity. The unit should be 'kg'.
        After the truck is full, the state will change to pending.
        '''
        self.product = product
        self.weight = self.capacity
        self.state = 'pending for delivery'
        self.operable_flag = True

    def unload_cargo(self) -> None:
        '''
        Unload cargo. If truck is empty, state become waiting.
        '''
        self.weight = 0
        self.state = 'waiting'
        self.operable_flag = True
        self.last_transport += self.capacity

    def get_truck_state(self) -> int:
        if self.operable_flag:
            return 1
        else:
            return 0
        
    def get_truck_produce(self) -> int:
        if self.product.startswith('P'):
            return int(self.product[1:])
        else:
            return ord(self.product) - ord('A') + 45

class Factory(object):
    def __init__(self, factory_id:str = 'Factory0',\
                 rate:float = 0.1,\
                 material:list = ['P1'],\
                 product: dict = {'P1':{}}) -> None:
        self.id = factory_id
        self.rate = rate
        self.total_final_product = 0
        self.warehouse = pd.DataFrame(columns=['Material', 'Quantity'], dtype=float)
        self.warehouse.set_index('Material', inplace=True)
        for tmp_material in material:
            self.add_material(tmp_material)
        self.production_requirements = product
        for tmp_product in product.keys():
            self.add_material(tmp_product)
    
    def add_material(self, material:str = None, quantity:float = 0.0) -> None:
        '''Add raw material or product to the warehouse.'''
        if material in self.warehouse.index:
            self.warehouse.loc[material, 'Quantity'] += quantity
        else:
            self.warehouse.loc[material] = quantity
    
    def produce(self) -> None:
        '''Produce products by consuming raw materials.'''
        for tmp_product, tmp_raw in self.production_requirements.items():
            item_num = 0
            if len(tmp_raw.keys()) == 0:
                # Produce raw materials
                item_num = self.rate
                self.add_material(tmp_product, self.rate)
            else:
                # Produce the product
                item_num = min(self.rate, min(self.warehouse.loc[tmp_raw.keys(), 'Quantity'] / list(tmp_raw.values())))
                self.add_material(tmp_product, item_num)
                # Consume material
                for tmp_item, tmp_ratio in tmp_raw.items():
                    self.add_material(tmp_item, -item_num*tmp_ratio)
            if tmp_product == 'A' or tmp_product == 'B' or tmp_product == 'C' or tmp_product == 'D' or tmp_product == 'E':
                self.total_final_product += item_num
    
    def load_truck(self, truck:Truck, product:str) -> None:
        '''Load product onto truck if the truck is in the 'waiting' state and enough product is available.'''
        capcity = truck.capacity
        if self.warehouse.loc[product,'Quantity'] >= capcity:
            truck.load_cargo(product)
            self.add_material(product, -capcity)
    
    def unload_truck(self, truck:Truck) -> None:
        '''Unload all cargo from the truck.'''
        if truck.product in self.warehouse.index:
            capcity = truck.capacity
            truck.unload_cargo()
            self.add_material(truck.product, capcity)

class Producer(object):
    '''
    product new product, load cargo, etc.
    '''
    def __init__(self, factory:list[Factory], truck:list[Truck], transport_idx: dict) -> None:
        self.factory = factory
        self.truck = truck
        self.transport_idx = transport_idx
        # Generate product list for loading
        material_list = [f'P{i}' for i in range(45)]
        self.product_idx = {f'Factory{index}': mat for index, mat in enumerate(material_list)}
        for i in range(5):
            self.product_idx[f'Factory{i+45}'] = None
    
    def produce_all(self) -> None:
        '''Produce product at all factory'''
        [tmp_factory.produce() for tmp_factory in self.factory]

    def unload_all(self) -> None:
        '''unload product at all factory'''
        for tmp_truck in self.truck:
            if tmp_truck.state == 'arrived':
                factory_id = tmp_truck.position
                id_idx = int(factory_id[7:])
                self.factory[id_idx].unload_truck(tmp_truck)

    def load_all(self) -> None:
        '''load product for all avaiable truck'''
        for tmp_truck in self.truck:
            if tmp_truck.state == 'waiting':
                factory_id = tmp_truck.position
                id_idx = int(factory_id[7:])
                tmp_product = self.product_idx[factory_id]
                if tmp_product is not None:
                    self.factory[id_idx].load_truck(tmp_truck, tmp_product)