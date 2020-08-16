# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 17:08:59 2020

@author: skitr
"""
import numpy as np
import scipy.sparse as sp
from scipy.integrate import RK45
import suport_functions
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from matplotlib import animation


n=160;
L=5

x=np.linspace(0,L,n);
y=np.linspace(0,L,n);
(xx,yy)=np.meshgrid(x,y)

pointsx1=range(n//7-1,int(n/2.5)-1);
pointsx=range(n//7,int(n/2.5));
pointsx2=range(n//7+1,int(n/2.5)+1);
pointsy=[n//2] * len(pointsx)



vx=np.zeros((n,n));
vx[pointsx1,pointsy]=0.025;
vx[pointsx,pointsy]=0.05;
vx[pointsx2,pointsy]=0.025;

Bvx=np.logical_or(xx==0,xx==L)
Bvx[pointsx,pointsy]=True;

vy=np.zeros((n,n));
vy[pointsx,pointsy]=0.012;
Bvy=np.logical_or(yy==0,yy==L)
Bvy[pointsx,pointsy]=True;


de=np.ones((n,n))*1.2754
Bde=np.zeros((n, n), dtype=bool)


u0=np.concatenate([np.reshape(vx,n*n), np.reshape(vy,n*n), np.reshape(de,n*n) ])

B=np.concatenate([np.reshape(Bvx,n*n), np.reshape(Bvy,n*n), np.reshape(Bde,n*n) ])


#−1/2	0	1/2

l=np.ones(n)
M=sp.spdiags([l*-0.5, l*0.5], [-1,1], n, n).todense();
M[0,0:3]=[-3/2,	2,	-1/2];
M[n-1,n-3:n]=[1/2,	-2,	3/2]
M=sp.csr_matrix(M);



(Dx, Dy)=suport_functions.getDxDy(M);
Dx=Dx/(x[1]-x[0])
Dy=Dy/(y[1]-y[0])

def uDerivative(t,u):
    u=np.reshape(u,(3,n*n))
    vx=u[0,]
    vy=u[1,]
    de=u[2,]
    
    dde=-Dx@(de*vx)-Dy@(de*vy)  
    
    #ideal gas equation
    c=0.286*(273+26)
    
    dvx=-(dde*vx+c*Dx@de+Dx@(de*vx*vx)+Dy@(de*vx*vy))/de
    
    dvy=-(dde*vy+c*Dy@de+Dy@(de*vy*vy)+Dx@(de*vx*vy))/de
    
    
    du=np.concatenate([dvx,dvy,dde])
    
    du[B]=0;
    return du

du=uDerivative(0,u0)


u=u0;
sol=RK45(uDerivative,0,u,100,vectorized=False)



#sol.h_abs = 0.0001

t_list=[]

save_folder='video_frames1'

n_new=18;
plot_indexs=np.rint(np.linspace(0,n-1,n_new)).astype(int)
(plot_indexsxx,plot_indexsyy)=np.meshgrid(plot_indexs,plot_indexs)
plot_indexsyy=np.reshape(plot_indexsyy,n_new*n_new)
plot_indexsxx=np.reshape(plot_indexsxx,n_new*n_new)

fig, ax = plt.subplots()
fig.set_size_inches(10.5, 10.5, forward=True)
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
plt.tight_layout()

up=np.reshape(u,(3,n*n))
up=np.reshape(up,(3,n,n))
Q = ax.quiver(xx[plot_indexsxx,plot_indexsyy],yy[plot_indexsxx,plot_indexsyy],up[0,plot_indexsxx,plot_indexsyy],up[1,plot_indexsxx,plot_indexsyy],scale=1)

t=0;

num_p=50;
pa_xx=np.random.uniform(low=0.5, high=4.5, size=(num_p))
pa_yy=np.random.uniform(low=0.5, high=4.5, size=(num_p))

#sc=ax.scatter(pa_xx,pa_yy)

def update_quiver(num):
    """updates the horizontal and vertical vector components by a
    fixed increment on each frame
    """

    sol.step()
    print('t:',sol.t)
    u=sol.y
    up=np.reshape(u,(3,n*n))
    up=np.reshape(up,(3,n,n))
    
    global pa_xx
    global pa_yy
    
    
    pa_vx_i = np.floor(pa_xx/L*n).astype(int)
    pa_vy_i = np.floor(pa_yy/L*n).astype(int)
    
    pa_vx_i[pa_vx_i>=n]=n-1
    pa_vy_i[pa_vy_i>=n]=n-1
    
    pa_vx_i = list(pa_vx_i)
    pa_vy_i = list(pa_vy_i)

    pa_xx=pa_xx+1000*sol.h_abs*up[0,pa_vx_i,pa_vy_i]
    pa_yy=pa_yy+1000*sol.h_abs*up[1,pa_vx_i,pa_vy_i]
    
    
    data = np.hstack((pa_xx[:,np.newaxis], pa_yy[:, np.newaxis]))
    #sc.set_offsets(data)
    
    Q.set_UVC(up[0,plot_indexsxx,plot_indexsyy],up[1,plot_indexsxx,plot_indexsyy])
    
    return Q,


anim = animation.FuncAnimation(fig, update_quiver,blit=False, interval=0.5,
                              repeat=True, save_count=800)

anim.save("basic_animation.mp4",fps=60)
plt.show()

    
