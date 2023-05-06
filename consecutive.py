
import numpy as np
from mpi4py import MPI
import copy
import random
import matplotlib.pyplot as plt

class Particle:
    def __init__(self,interval, xna, data,user,appliance,Ph_forecast):
        self.xna = xna.copy()
        self.data=copy.deepcopy(data)
        self.velocity = 0
        self.best_xna = xna.copy()
        self.best_fitness = float('inf')
        self.user=user
        self.appliance=appliance
        self.Ph_forecast=Ph_forecast
        self.interval=interval

    def update_velocity(self,  inertia_weight, c1, c2,global_best_xna):
        r1 = np.round(np.random.rand(),2)
        r2 =np.round(np.random.rand(),2)
   
        cognitive_velocity = c1 * r1 *(np.round(np.sum(self.best_xna))-np.round(np.sum(self.xna)))
        social_velocity = c2 * r2 *(np.round(np.sum(global_best_xna))-np.round(np.sum(self.xna)))
        self.velocity = inertia_weight * self.velocity + cognitive_velocity + social_velocity
        self.velocity=((int(self.velocity)//10)+1)%H
        

    def update_xna(self):
        new_indices = np.array([int(i) for i in range(0,H,self.velocity)])
        temp={}
        for i in new_indices:
          start=self.interval[user][appliance][0]
          end=self.interval[user][appliance][1]
          if i >= start and i<=end:
            temp[i]=self.xna[i]
        values = list(temp.values())
        random.shuffle(values)
        shuffled_dict = {k: v for k, v in zip(temp.keys(), values)}
        for x,y in shuffled_dict.items():
          self.xna[x]=y    
        self.data[self.user][self.appliance]=self.xna  

    def update_best_xna(self, fitness):
        if fitness < self.best_fitness:
            self.best_xna = self.xna.copy()
            self.best_fitness = fitness

    def calculateLh(self,ln,l_n):
      Lh = ln
      for l in l_n:
        Lh = [round(x + y,2) for x, y in zip(Lh, l)]
      return Lh

    def payoff(self,ln, l_n, Lnh):
        k=10;a=1;b=2
        Lh=self.calculateLh(ln,l_n)
        Ph_forecast=self.Ph_forecast
        bh=[]
        for i in range(H):
            if(Lh[i]<=Ph_forecast[i]):
                bh.append(k)
            else:
                bh.append(k+a*(np.square((Lh[i]-Ph_forecast[i])/Ph_forecast[i]))+b*((Lh[i]-Ph_forecast[i])/Ph_forecast[i]))
        Bn=np.sum(np.dot(bh,Lnh[self.user]))
        return Bn

    def evaluate_fitness(self):
      Lnh={}
      for user in self.data:
        Lnh[user]=[0]*H
        for appliance in self.data[user]:
          Lnh[user]=[round(x + y,2) for x, y in zip(Lnh[user], self.data[user][appliance])]
      
      ln=Lnh[self.user]
      l_n=[Lnh[x] for x in Lnh if x!=self.user]
      return self.payoff(ln,l_n,Lnh)

  
class PSO:
    def __init__(self,interval,permutations, num_particles, num_iterations,  xna,data, inertia_weight, c1, c2,  user, appliance,Ph_forecast):
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.data = data
        self.xna = xna
        self.inertia_weight = inertia_weight
        self.c1 = c1
        self.c2 = c2
        self.user = user
        self.appliance = appliance
        self.particles = []
        self.global_best_xna = xna.copy()
        self.global_best_fitness = float('inf')
        self.Ph_forecast=Ph_forecast
     
        self.permutations=copy.deepcopy(permutations)
        self.interval=interval

    def initialize_particles(self):
        for i in range(self.num_particles):
            self.xna=self.permutations[i]
            self.data[self.user][self.appliance] = self.xna
            particle = Particle(self.interval,self.xna, self.data,  self.user,self.appliance,self.Ph_forecast)
            self.particles.append(particle)

    def run(self):
        self.initialize_particles()
        for i in range(self.num_iterations):
            for particle in self.particles:
                if i!=0:
                  particle.update_velocity(self.inertia_weight, self.c1, self.c2, self.global_best_xna)
                  particle.update_xna()
                fitness = particle.evaluate_fitness()
                
                particle.update_best_xna(fitness)

                if fitness < self.global_best_fitness:
                    self.global_best_xna = particle.xna.copy()
                    self.global_best_fitness = fitness
        return self.global_best_xna, self.global_best_fitness
    

#No of hours
H = 5
#min and max value of xhn,a
min_value = 2
max_value = 8
#no of appliances of each user
num_appliances = 3
#no of users
num_users=10

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

num_iterations = 10
inertia_weight = 0.7
c1 = 0.2
c2 = 0.3

def generate_values(start,end,target,period):
  values = []  
  for i in range(period):
    values.append(round(target/period,2))
  values=values+[0]*(end-start-period+1)
  return values

if(rank==0):
  ####uncomment this to give random values####
  # interval={}
  # for user in range(num_users):
  #   appliances={}
  #   for appliance in range(num_appliances):
  #     alpha=0
  #     beta=0
  #     period=round(random.randint(1, 4))
  #     while alpha>=beta or beta-alpha<period:
  #       alpha = round(random.randint(0, H-1))
  #       beta = round(random.randint(0, H-1))    
  #     Ea=round(random.randint(10, 50))
  #     appliances[f"appliance{appliance+1}"] = [alpha,beta,Ea,period]
  #   interval[f"user{user+1}"] = appliances
  interval = {"user1":{"appliance1": [1, 3, 24, 2],"appliance2": [0, 1, 44, 1],"appliance3": [1, 4, 25, 3]},
            "user2":{"appliance1": [1, 3, 27, 2],"appliance2": [1, 4, 43, 2],"appliance3": [1, 4, 27, 3]},
            "user3":{"appliance1": [0, 3, 13, 2],"appliance2": [1, 2, 49, 1],"appliance3": [2, 4, 16, 2]},
            "user4":{"appliance1": [0, 4, 16, 2],"appliance2": [1, 3, 22, 1],"appliance3": [0, 2, 27, 1]},
            "user5":{"appliance1": [0, 3, 48, 2],"appliance2": [0, 4, 33, 3],"appliance3": [0, 4, 24, 4]},
            "user6":{"appliance1": [0, 2, 38, 1],"appliance2": [0, 4, 40, 4],"appliance3": [0, 4, 38, 2]},
            "user7":{"appliance1": [0, 2, 44, 2],"appliance2": [0, 3, 36, 3],"appliance3": [1, 3, 42, 1]},
            "user8":{"appliance1": [0, 2, 33, 2],"appliance2": [0, 4, 39, 4],"appliance3": [0, 4, 30, 4]},
            "user9":{"appliance1": [1, 4, 37, 3],"appliance2": [0, 3, 37, 2],"appliance3": [0, 4, 17, 4]},
            "user10":{"appliance1": [1, 2, 45, 1],"appliance2": [0, 4, 27, 2],"appliance3": [0, 4, 25, 3]},
  }
  for user in interval:
    print(f"{user}:")
    for appliance in interval[user]:
      print(f"{appliance}:",end=" ")
      print(interval[user][appliance])

  comm.bcast(interval, root=0)
else:
  interval = comm.bcast(None, root=0)

comm.barrier()

if(rank==0):
  energy = {}
  for user in interval:
    appliances = {}
    for appliance in interval[user]:
        start=interval[user][appliance][0]
        end=interval[user][appliance][1]
        values=generate_values(start,end,interval[user][appliance][2],interval[user][appliance][3])
        values=[0]*start+values+[0]*(H-end-1)
        appliances[appliance] = values
    energy[user] = appliances

  for user in energy:
      print(f"{user}:")
      for appliance in energy[user]:
        print(f"{appliance}:",end=" ")
        print(energy[user][appliance])
  comm.bcast(energy, root=0)
else:
    energy = comm.bcast(None, root=0)


wuslm=[0]*H
for user in energy:
  for appliance in energy[user]:
    wuslm=np.add(wuslm,energy[user][appliance])
peak_before=max(wuslm)
avg_before=sum(wuslm)/len(wuslm)
par_before=peak_before/avg_before

xna=[]
users_per_process = num_users // size
if(num_users!=size):
   print("Number of process should be equal to number of uers")
   exit(0)

start_user = rank * users_per_process
end_user = start_user + users_per_process
users=list(energy.keys())

if(rank==0):
  Ph_forecast_RE=[round(random.uniform(5,17),2) for _ in range(H)]
  Ph_diesel=[round(random.uniform(5,17),2) for _ in range(H)]
  Ph_forecast=np.add(Ph_forecast_RE,Ph_diesel)
  print(Ph_forecast_RE,Ph_diesel)
  comm.bcast(Ph_forecast, root=0)
else:
  Ph_forecast = comm.bcast(None, root=0)
    

start_time=MPI.Wtime()

def generate_combinations(lst):
    n = len(lst)
    val =lst[0]
    num_ones = sum(1 for x in lst if x == val)
    result = []
    for i in range(n - num_ones + 1):
        lst[:i]=[0]*i
        p =lst[:i] + [val] * num_ones + lst[i+num_ones:]
        result.append(list(p))
    
    return result


for user in users[start_user:end_user]:
  permutations={}
  for appliance in energy[user]:
    permutations[appliance]=[ ]
    start=interval[user][appliance][0]
    end=interval[user][appliance][1]
    num_particles=end-start-interval[user][appliance][3]+2
  
    x=copy.deepcopy(energy[user][appliance])
    y=x[start:end+1]
    p=generate_combinations(y)
    for i in p:
      x[start:end+1]=i
      permutations[appliance].append(x)

  completed=False
  update=False
 
  while True:
    prev=copy.deepcopy(energy)#.copy()
    x=copy.deepcopy(energy)
    for appliance in energy[user]:
      start=interval[user][appliance][0]
      end=interval[user][appliance][1]
      num_particles=end-start-interval[user][appliance][3]+2
      if update==False:
        xna=energy[user][appliance]
        
        pso = PSO(interval,permutations[appliance],num_particles, num_iterations, xna, energy, inertia_weight, c1, c2, user,appliance,Ph_forecast)
        
        pso.run()
     
        global_best_xna = pso.global_best_xna
        global_best_fitness = pso.global_best_fitness
        energy[user][appliance]=global_best_xna

    prev=copy.deepcopy(energy)
    flag=True
    for i in range(size):
      if i == rank:
        comm.bcast((energy[user],user),root=rank)
      else:
        message=comm.bcast(None,root=i)
        energy[message[1]]=message[0]

    update=True
    for u in energy:
      if energy[u]!=x[u] and user!=u:        
        update=False
        break
    uslm=[0]*H
    for usr in energy:
      for appliance in energy[usr]:
        uslm=np.add(uslm,energy[usr][appliance])
    
    peak_after=max(uslm)
    avg_after=sum(uslm)/len(uslm)
    par_after=peak_after/avg_after
    if par_before>par_after:
      completed=True
    else:
      completed=False
    comm.barrier()
    all_comp = comm.allgather(completed)
    comm.Barrier()
    flag = all(x for x in all_comp)

    if flag:
      break

end_time=MPI.Wtime()

if(rank==0):
  print("Time Taken",end_time-start_time)
  print("Final schedules")
  for user in energy:
    print(f"{user}:")
    for appliance in energy[user]:
      print(f"{appliance}:",end=" ")
      print(energy[user][appliance])
  uslm=[0]*H
  for user in energy:
    for appliance in energy[user]:
      uslm=np.add(uslm,energy[user][appliance])
  
  peak_after=max(uslm)
  avg_after=sum(uslm)/len(uslm)
  par_after=peak_after/avg_after
  print(peak_before,peak_after)

  fig,ax=plt.subplots(figsize=(10,8))
  ax.plot(range(len(uslm)), uslm,color="r",label='uslm')
  ax.plot(range(len(wuslm)),wuslm,color="b",label='wuslm')
  ax.legend()
  plt.xticks(range(len(uslm)))
  plt.show()
MPI.Finalize()