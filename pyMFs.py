import numpy as np 
import sys
from sympy import LeviCivita
from numba import jit

@jit
def CalculateMFs(data,is_periodic=False,deltabin=0.3,is_need_calculate_bin=True):
    data=np.array(data)
    datashape=np.shape(data)
    #print(datashape)
    HII_x=datashape[0]
    HII_y=datashape[1]
    HII_z=datashape[2]
    volume=HII_x*HII_y*HII_z
    v0=np.zeros((61,),dtype=np.float32)
    v1=np.zeros((61,),dtype=np.float32)
    v2=np.zeros((61,),dtype=np.float32)
    v3=np.zeros((61,),dtype=np.float32)
    gradnorm=np.zeros((HII_x,HII_y,HII_z),dtype=np.float64)
    sum1=np.zeros((HII_x,HII_y,HII_z),dtype=np.float64)
    sum2=np.zeros((HII_x,HII_y,HII_z),dtype=np.float64)
    
    levicivita=np.zeros((3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                levicivita[i][j][k]=LeviCivita(i,j,k)
    levicivita=np.array(levicivita,dtype=np.float32)
    sig=np.std(data)
    if is_need_calculate_bin:
        deltabin=deltabin*sig
    dth=0.5*deltabin
    if not is_periodic:
        grad,secondgrad=hessian(data)
        grad=np.array(grad).transpose(1,2,3,0)
        secondgrad=np.array(secondgrad).transpose(2,3,4,0,1)
        s=0
        for i in range(HII_x):
            for j in range(HII_y):
                for k in range(HII_z):
                    s+=1
                    sum1temp=0.0
                    sum2temp=0.0
                    gradtemp=grad[i][j][k]
                    secondgradtemp=secondgrad[i][j][k]
                    gradnorm[i][j][k]=np.linalg.norm(gradtemp)
                    #print(secondgrad)
                    if (gradnorm[i][j][k]==0):
                        sum1[i][j][k]=0
                        sum2[i][j][k]=0
                        continue

                    for l in range(3):
                        for m in range(3):
                            if l==m:
                                continue
                            sum1temp+=gradtemp[l]*secondgradtemp[m][0]*(levicivita[l][m][2]*gradtemp[1]-levicivita[l][m][1]*gradtemp[2])
                            sum1temp+=gradtemp[l]*secondgradtemp[m][1]*(levicivita[l][m][0]*gradtemp[2]-levicivita[l][m][2]*gradtemp[0])
                            sum1temp+=gradtemp[l]*secondgradtemp[m][2]*(levicivita[l][m][1]*gradtemp[0]-levicivita[l][m][0]*gradtemp[1])
                            for n in range(3):
                                sum2temp+=levicivita[l][m][n]*gradtemp[l]*gradtemp[0]*(secondgradtemp[m][1]*secondgradtemp[m][2]-secondgradtemp[m][2]*secondgradtemp[n][1])
                                sum2temp+=levicivita[l][m][n]*gradtemp[l]*gradtemp[1]*(secondgradtemp[m][2]*secondgradtemp[m][0]-secondgradtemp[m][0]*secondgradtemp[n][2])
                                sum2temp+=levicivita[l][m][n]*gradtemp[l]*gradtemp[2]*(secondgradtemp[m][0]*secondgradtemp[m][1]-secondgradtemp[m][1]*secondgradtemp[n][0])
                    if np.isnan(sum1temp):
                        sum1[i][j][k]=0.0
                    else:
                        sum1[i][j][k]=sum1temp
                    if np.isnan(sum2temp):
                        sum2[i][j][k]=0.0
                    else:
                        sum2[i][j][k]=sum2temp
                    #print(s)
        sum1=sum1/(gradnorm**2)
        sum2=sum2/(2*gradnorm**3)
        print('start loop')
        for i in range(61):
            v0[i],v1[i],v2[i],v3[i]=calculateKdr(data,gradnorm,sum1,sum2,(i-30)*0.1*sig,deltabin,HII_x,HII_y,HII_z,volume,dth,levicivita)
    return v0,v1,v2,v3
@jit
def calculateKdr(data,gradnorm,sum1,sum2,threshold,deltabin,HII_x,HII_y,HII_z,volume,dth,levicivita):
    n=0.0
    v1=0.0
    v2=0.0
    v3=0.0
    for i in range(HII_x):
        for j in range(HII_y):
            for k in range(HII_z):
                if data[i][j][k]>=threshold:
                    n=n+1.0
                if (np.abs(data[i][j][k]-threshold)<dth):
                    #gradcache=grad[:,i,j,k]
                    #secondgradcache=secondgrad[:,:,i,j,k]
                    v1+=gradnorm[i][j][k]
                    v2+=sum1[i][j][k]
                    v3+=sum2[i][j][k]
                #l+=1
                #print(l)
    v0=n/volume
    v1=v1/(6*volume*deltabin)
    v2=v2/(6*volume*deltabin*np.pi)
    v3=v3/(4*volume*deltabin*np.pi)
    return v0,v1,v2,v3



def hessian(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x) 
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype) 
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k) 
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return x_grad,hessian



def calc_mfs_integrant(grad,secondgrad,LeviCivita):
    gradnorm=np.linalg.norm(grad)
    #print(secondgrad)
    if (gradnorm==0):
        return 0.0,0.0,0.0
    sum1=0.0
    sum2=0.0

    for i in range(3):
        for j in range(3):
            if i==j:
                continue
            sum1+=grad[i]*secondgrad[j][0]*(LeviCivita[i][j][2]*grad[1]-LeviCivita[i][j][1]*grad[2])
            sum1+=grad[i]*secondgrad[j][1]*(LeviCivita[i][j][0]*grad[2]-LeviCivita[i][j][2]*grad[0])
            sum1+=grad[i]*secondgrad[j][2]*(LeviCivita[i][j][1]*grad[0]-LeviCivita[i][j][0]*grad[1])
            for k in range(3):
                sum2+=LeviCivita[i][j][k]*grad[i]*grad[0]*(secondgrad[j][1]*secondgrad[j][2]-secondgrad[j][2]*secondgrad[k][1])
                sum2+=LeviCivita[i][j][k]*grad[i]*grad[1]*(secondgrad[j][2]*secondgrad[j][0]-secondgrad[j][0]*secondgrad[k][2])
                sum2+=LeviCivita[i][j][k]*grad[i]*grad[2]*(secondgrad[j][0]*secondgrad[j][1]-secondgrad[j][1]*secondgrad[k][0])
    sum1=np.float(sum1)
    sum2=np.float(sum2)
    if np.isnan(sum1):
        sum1=0.0
    if np.isnan(sum2):
        sum2=0.0
    sum1=sum1/(gradnorm**2)
    sum2=sum2/(2*gradnorm**3)
    return gradnorm,sum1,sum2

