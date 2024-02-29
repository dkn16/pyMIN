import numpy as np 


from sympy import LeviCivita
from numba import jit


@jit
def calculateMFs(data,thresholds=[],min_sig=-3,max_sig=3,step=0.1,deltabin=0.4,is_need_calculate_bin=True):
    '''
    Calculate the 3D MFs (v_{0}-v_{3}) of a given field (must be 3D)

    Parameters: 
        data: the field to be calculated
        threshdold: the threshold values to calculate MFs. if not given, it will be from min_sig*std(data) to max_sig*std(data) with a step size of *step* *std(data)
        min_sig: only take effect if threshold is not given. min_sig*std(data) is the lower bound of threshold values.
        max_sig: only take effect if threshold is not given. max_sig*std(data) is the upper bound of threshold values.
        step: only take effect if threshold is not given. step*std(data) is the step size of threshold values.
        delta_bin: in numeric calculation we treat the delta function as a window function. delta_bin is the width of the window function. usually no need to modify.
        is_need_calculate_bin: turn on if delta_bin represents a multiple of std(field)

    Returns: 
        v0,v1,v2,v3:  (v_{0}-v_{3}) of the given field
    '''
    data=np.array(data,dtype=np.float64)
    datashape=np.shape(data)
    sig=np.std(data)
    #print(datashape)
    HII_x=datashape[0]
    HII_y=datashape[1]
    HII_z=datashape[2]
    volume=HII_x*HII_y*HII_z
    if len(thresholds)<1:
        thresholds=np.linspace(min_sig,max_sig,int((max_sig-min_sig)/step)+1)*sig
    nums=len(thresholds)
    v0=np.zeros((nums,),dtype=np.float32)
    v1=np.zeros((nums,),dtype=np.float32)
    v2=np.zeros((nums,),dtype=np.float32)
    v3=np.zeros((nums,),dtype=np.float32)
    gradnorm=np.zeros((HII_x,HII_y,HII_z),dtype=np.float64)
    sum1=np.zeros((HII_x,HII_y,HII_z),dtype=np.float64)
    sum2=np.zeros((HII_x,HII_y,HII_z),dtype=np.float64)
    
    levicivita=np.zeros((3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                levicivita[i][j][k]=LeviCivita(i,j,k)
    levicivita=np.array(levicivita,dtype=np.float32)
    
    if is_need_calculate_bin:
        deltabin=deltabin*sig
    dth=0.5*deltabin

    grad,secondgrad=hessian(data)
    grad=np.array(grad).transpose(1,2,3,0)
    secondgrad=np.array(secondgrad).transpose(2,3,4,0,1)
    gradnorm=np.linalg.norm(grad,axis=3)
    s=0
    for i in range(HII_x):
        for j in range(HII_y):
            for k in range(HII_z):
                s+=1
                sum1temp=0.0
                sum2temp=0.0
                gradtemp=grad[i][j][k]
                secondgradtemp=secondgrad[i][j][k]
                #gradnorm[i][j][k]=np.linalg.norm(gradtemp)
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
                            sum2temp+=levicivita[l][m][n]*gradtemp[l]*gradtemp[0]*(secondgradtemp[m][1]*secondgradtemp[n][2]-secondgradtemp[m][2]*secondgradtemp[n][1])
                            sum2temp+=levicivita[l][m][n]*gradtemp[l]*gradtemp[1]*(secondgradtemp[m][2]*secondgradtemp[n][0]-secondgradtemp[m][0]*secondgradtemp[n][2])
                            sum2temp+=levicivita[l][m][n]*gradtemp[l]*gradtemp[2]*(secondgradtemp[m][0]*secondgradtemp[n][1]-secondgradtemp[m][1]*secondgradtemp[n][0])
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
    mask=np.isnan(sum1)
    sum1[mask]=0
    mask=np.isnan(sum2)
    sum2[mask]=0
    mask=np.where(np.abs(sum2)>1000)
    sum2[mask]=0
    #print('start loop')
    
    i=0
    for j in thresholds:
        v0[i],v1[i],v2[i],v3[i]=calculateKdr(data,gradnorm,sum1,sum2,j,deltabin,HII_x,HII_y,HII_z,volume,dth)
        i+=1
    return v0,v1,v2,v3

@jit
def calculateKdr(data,gradnorm,sum1,sum2,threshold,deltabin,HII_x,HII_y,HII_z,volume,dth):
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





import math
@jit
def analyticalMFs(data,thresholds=[]):
    """
    Calculate the analytical MFs of a given field according to the eqns.
    Parameters:
        data: the field to be calculated
        threshdold: the threshold values to calculate MFs. if not given, it will be from -3*std(data) to 3*std(data) with a step size of 0.1 *std(data)
    Returns:
        v0,v1,v2,v3: analytical (v_{0}-v_{3}) of the given field

    Note that this analytical equation only valid when the field is Gaussian.
    """
    datashape=np.shape(data)
    if len(thresholds)<1:
        thresholds=np.linspace(-3,3,61)
    #print(datashape)
    HII_x=datashape[0]
    HII_y=datashape[1]
    HII_z=datashape[2]
    v=HII_x*HII_y*HII_z
    data=data-np.mean(data)
    sigma=np.sqrt(np.sum(data**2/v))
    gradient=np.array(np.gradient(data)).transpose(1,2,3,0)
    norm=np.linalg.norm(gradient,axis=3)
    sigma1=np.sqrt(np.sum(norm**2/v))
    lam=math.sqrt(1/(6*math.pi))*(sigma1/sigma)
    #print(lam)
    v0=np.zeros(len(thresholds))
    v1=np.zeros(len(thresholds))
    v2=np.zeros(len(thresholds))
    v3=np.zeros(len(thresholds))
    for i in range(len(thresholds)):
        v0[i]=math.erfc(thresholds[i]/math.sqrt(2))/2
        v1[i]=(2*lam)*(math.e**(-0.5*thresholds[i]**2))/(3*math.sqrt(2*math.pi))
        v2[i]=(2*lam**2)/(3*math.sqrt(2*math.pi))*(math.e**(-0.5*thresholds[i]**2))*thresholds[i]
        v3[i]=(lam**3)/(math.sqrt(2*math.pi))*(math.e**(-0.5*thresholds[i]**2))*(thresholds[i]**2-1)
    return v0,v1,v2,v3

@jit
def substractwedge(data,m=0.5,nu_axis=2):
    if nu_axis==2 :
        x=np.shape(data)[0]
    else:
        x=np.shape(data)[2]
    z=np.shape(data)[nu_axis]
    datak=np.fft.rfftn(data)
    x=np.fft.fftfreq(x)
    z=np.fft.rfftfreq(z)
    kx,ky,kz=np.meshgrid(x,x,z)
    k_para=np.sqrt(kx**2+ky**2)
    k_perp=kz
    mask=np.where(k_para<m*k_perp)
    datak[mask]=0
    return np.fft.irfftn(datak)
