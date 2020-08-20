import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np

fig, ax = plt.subplots()
x, y = [],[]
ax.grid()
sc = ax.scatter(x,y)
plt.xlim(0,10)
plt.ylim(0,10)

def animate(i):
    a, b = [], []
    print(type(i))
    a.append(np.random.rand(1)*10)
    b.append(np.random.rand(1)*10)
    sc.set_offsets(np.c_[a,b])

ani = matplotlib.animation.FuncAnimation(fig, animate, 
                frames=10, interval=1000, repeat=True) 
plt.show()




import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np

fig, ax = plt.subplots()  
ax.grid()  

data = np.cumsum(np.random.normal(size=100)) #some list of data
s, = ax.plot(data[::2], data[1::2], marker="o", ls="") # set linestyle to none

ss = []

def plot(a, data):
    
    ax.clear()
    
    sc, = ax.plot(data[::2], data[1::2], marker="o", ls="") # set linestyle to none    
    ss.append(sc)
    
    data += np.cumsum(np.random.normal(size=100)+3e-2)
    sc.set_data(data[::2], data[1::2])
    ax.relim()
    ax.autoscale_view(True,True,True)
    
ani = matplotlib.animation.FuncAnimation(fig, plot, fargs=(data,),
            frames=4, interval=100, repeat=True) 
plt.show()