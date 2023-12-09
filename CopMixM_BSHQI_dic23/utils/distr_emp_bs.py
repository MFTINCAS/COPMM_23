# -*- coding: utf-8 -*-

import numpy as np
import scipy.interpolate as spint
import matplotlib.pylab as plt
import scipy
from KDEpy import FFTKDE
from utils.ISJ_bandwidth import improved_sheather_jones
from sklearn.model_selection import cross_val_score


def distempcont_bs2(Xd,wd):
    """
    Esempio di distribuzione empirica continua
    """
    
    Nd = len(Xd)

   
    Ne  = 1 + int( np.log(Nd))
    Ne =  1 + int( (Nd**(1/3)))#scelta bandwidth
  
    # Scott
    # hb=int(3.5*np.std(Xd))/(Nd**(1/3)) 
    # print('hb_bsqi',hb)
    # Xdm = min(Xd)
    # XdM = max(Xd)
    # Ne=int(np.floor((XdM-Xdm)/hb))
    
     
       #Silvermean 
       # Calcola l'intervallo interquartile (IQR)
       # q75, q25 = np.percentile(Xd, [75 ,25])
       # IQR = q75 - q25

       # # Numero di osservazioni

       # # Calcola la larghezza della banda con la regola di Silverman
       # h_silverman = 0.9 * min(np.std(Xd) / 1.34, (IQR / (1.34 * Nd**(-1/5))))
       # Ne=int(np.floor((XdM-Xdm)/h_silverman))
       

       #Ne=4
      
    if Ne%2 == 0:
       Ne = Ne+1
       #opt_max_EMP['distr_emp_bs2_bw'] = Ne


    
    ecdf_max=True
    warn_ne=False
    while ecdf_max :
        Nd = len(Xd)
        
        Xdm = min(Xd)
        XdM = max(Xd)

        #------------------------------------
        h=(XdM-Xdm)*1e-1#Mazzia
        #h=np.copy(hb)*1e-2
        
        ix = np.linspace(Xdm-h,XdM+h,Ne)
        ix = np.sort(ix)
        Ne = len(ix)
        pc = np.zeros(Ne)   
        for i in range(Ne):
            pc[i] = np.sum( wd[ Xd <= ix[i]] )#cumulata discreta
       
        if np.sum(wd)> 0:
           pc = pc/np.sum(wd)
        else:
           warn_ne=True 
       # ix =  np.concatenate(([ ix[0]-(ix[1]-ix[0]) ], ix,[  ix[Ne-1]+(ix[Ne-1]-ix[Ne-2]) ]),axis=0)
       # pc =  np.concatenate(([ 0 ], pc,[1]),axis=0)
       
        
       
        x = np.copy(ix) 
        y = np.copy(pc)
        
        
        
        
        
        
        
        Nx = x.shape[0]
    
        
        hx =  (x[1:(Nx)]-x[0:(Nx-1)]) 
        h=hx[0]
        #print('hx',hx)
        fy = (y[2:(Nx)]-y[0:(Nx-2)])/(hx[1:]+hx[0:(Nx-2)])
        diff=y[1:]-y[:-1]
        print("diff",diff)
        
        fy = np.concatenate(  ( [ (y[1]-y[0])/hx[0] ] ,fy,[(y[Nx-1]-y[Nx-2])/hx[Nx-2] ] ), axis=0 )   
        #cs = (y[1:]+y[0:(Nx-1)])/2.0 - (hx*(fy[1:]-fy[0:(Nx-1)]))/4.0
        #cs = np.concatenate(([y[0]],cs,[y[Nx-1]]),axis=0)
        #print('cs',np.sum(cs))
        xs = np.concatenate(([x[0]],[x[0]],x,[x[Nx-1]],[x[Nx-1]]),axis=0)
        #fyt2=scipy.interpolate.BSpline(xs, cs, 2, extrapolate=True, axis=0)
        
        
        fyy = (y[2:Nx]-2*y[1:(Nx-1)]+y[0:(Nx-2)])/(hx[1:]**2)
        #fyy = np.concatenate(  ( [ (fy[1]-fy[0])/hx[0] ] ,fyy,[(fy[Nx-1]-fy[Nx-2])/hx[Nx-2] ]  ), axis=0 )   
        fyy = np.concatenate(  ( [ 0 ] ,fyy,[ 0]  ), axis=0 )   
    
        csyy = (fy[1:]+fy[0:(Nx-1)])/2.0 - (hx*(fyy[1:]-fyy[0:(Nx-1)]))/4.0
        
        csyy = np.concatenate(([ fy[0] ],csyy,[ fy[Nx-1] ]),axis=0)
        #print('csyy',np.sum(csyy))
        xsyy = np.concatenate(([x[0]],[x[0]],x,[x[Nx-1]],[x[Nx-1]]),axis=0)#?? viene usata?
        
        fyydt2=scipy.interpolate.BSpline(xs, csyy, 2, extrapolate=True, axis=0)
        fyyt22 = scipy.interpolate.splantider(fyydt2)
        
        #controlli miei---------------------------------------------------
        sum1=np.sum(fyydt2.basis_element(ix)(ix))
        integral=np.trapz(fyydt2(xs), xs)
        #def uniform_kernel(X,x,h):
           # return (abs((X-x)/h) <= 0.5).astype(float)
        
        #integral=np.trapz(uniform_kernel(Xd, xs, h)*np.sum(basis_element(xs)(xs)), xs)
      
        
      
        
      
        
      
        #------------------------------------------------------------------
        # fd2 = scipy.interpolate.splder(fyt2,n=1)
        # y = fd2(x)
        # Nx = x.shape[0]
        
        # hx =  (x[1:(Nx)]-x[0:(Nx-1)])
        # fy = (y[2:(Nx)]-y[0:(Nx-2)])/(hx[1:]+hx[0:(Nx-2)])
        # fy = np.concatenate(  ( [ (y[1]-y[0])/hx[0] ] ,fy,[(y[Nx-1]-y[Nx-2])/hx[Nx-2] ] ), axis=0 )   
      
        # cs = (y[1:]+y[0:(Nx-1)])/2.0 + (hx*(fy[1:]-fy[0:(Nx-1)]))/4.0
        # cs = np.concatenate(([y[0]],cs,[y[Nx-1]]),axis=0)
        # xs = np.concatenate(([x[0]],[x[0]],x,[x[Nx-1]],[x[Nx-1]]),axis=0)
        # fd22=scipy.interpolate.BSpline(xs, cs, 2, extrapolate=True, axis=0)
        
        # fyt22 = scipy.interpolate.splantider(fd22)
        #epdf = fd22(Xd)
        #ecdf = fyt22(Xd)#/fyt22(np.max(Xd))
        epdf = fyydt2(Xd)
        #print('dist',np.sum(epdf)/len(x))
        yms = fyyt22(x)
        diff = yms-y
        dr = np.mean(diff,axis=0)
        drv=dr
        #print(drv)
        
        
        if fyyt22(XdM)-drv > 1.0001:
            
           ecdf = (fyyt22(Xd)-drv)/(fyyt22(XdM)-drv)

     
           
           ecdf_max=False      
           #Ne = Ne+1
           #opt_max_EMP['distr_emp_bs2_bw'] = Ne
           #print(Ne,fyyt22(XdM))
        else:
           ecdf = fyyt22(Xd)-drv
           ecdf_max=False
        #print('fyyt22',fyyt22(XdM+h))
        #print('fyyt22-drv',(fyyt22(XdM+h)-drv))
        #print('fyyt22picc',fyyt22(Xdm-h))
        #print('fyyt22-drvpicc',(fyyt22(Xdm-h)-drv))
      
    
    
    # plot=True
    # if plot:
    #     t = np.linspace(ix[0],ix[Ne-1],500)
    #     plt.figure(1)
    #     plt.plot(ix,pc,'.',t,fyt22(t),'--' )
    #     plt.show()
    #     plt.hist(Xd,bins=50,density=True)
    #     plt.plot(t,fd22(t),'--',t,fd2(t),'.' )
    #     plt.plot(t,fyydt2(t),'--',t,fd2(t),'.' )
    #     plt.show()
    
    
    return (epdf,ecdf,warn_ne)

    