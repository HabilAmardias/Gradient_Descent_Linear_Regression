import numpy as np
import math

def mse(y:np.ndarray,ycap:np.ndarray):
    assert y.shape==ycap.shape,"y and ycap must have same dimension"
    sum=np.sum(np.subtract(y,ycap)**2)
    return sum/len(ycap)

def mse_grad(y:np.ndarray,ycap:np.ndarray,exog:np.ndarray):
    assert len(y.shape)==1,"y must be 1 dimension"
    assert y.shape==ycap.shape,"y and ycap must have same dimension"
    assert len(exog.shape)<=2,"x must be either 1d or 2d"
    assert exog.shape[0]==y.shape[0],"x and y observation count are not same"
    grads_w=exog.T@np.subtract(ycap,y)
    grads_w=(2/len(y))*grads_w
    grads_b=(2/len(y))*np.sum(np.subtract(ycap,y))
    return grads_b,grads_w

def CosineScheduler(step:int,max_step:int,base_lr:float,final_lr:float):
    lr=final_lr+(base_lr-final_lr)*(1+math.cos(math.pi*step/max_step))/2
    return lr
class GDLinearRegression():
    def __init__(self,base_lr:float=3e-1,final_lr:float=1e-3,max_iter:int=100,verbose:int=1,cosine:bool=False):
        assert verbose==0 or verbose==1, "verbose must only 0 or 1"
        assert final_lr<base_lr, "Final LR must be smaller than Base LR"
        self.base_lr=base_lr
        self.final_lr=final_lr
        self.max_iter=max_iter
        self.verbose=verbose
        self.cosine=cosine
        self.params=None
    def fit(self,endog:np.ndarray,exog:np.ndarray):
        assert len(endog.shape)==1,"y must be 1 dimension"
        assert len(exog.shape)<=2,"x must be either 1d or 2d"
        assert exog.shape[0]==endog.shape[0],"x and y observation count are not same"
        if len(exog.shape)==1:
            W:np.ndarray=np.random.randn(2)
        else:
            W:np.ndarray=np.random.randn(exog.shape[1]+1)
        iter=0
        lr=self.base_lr
        while iter<self.max_iter:
            ycap=exog@W[1:]+W[0]
            grad_b,grad_w=mse_grad(endog,ycap,exog)
            W[1:]=W[1:]-lr*grad_w
            W[0]=W[0]-lr*grad_b
            ycap1=exog@W[1:]+W[0]
            if self.verbose==1:
                print(f'Iteration {iter+1} MSE with lr {lr}: ',mse(endog,ycap1))
            if self.cosine==True:
                lr=CosineScheduler(iter,self.max_iter,self.base_lr,self.final_lr)
            iter+=1
        self.params=W
    def predict(self,exog:np.ndarray):
        assert len(exog.shape)<=2,"x must be either 1d or 2d"
        assert exog.shape[1]+1==self.params.shape[0],"features doesn't match with params"
        return exog@self.params[1:]+self.params[0]

#testing     
if __name__=='__main__':
    X=np.random.rand(100,5)
    y=np.random.rand(100)
    model=GDLinearRegression(verbose=1,max_iter=10,cosine=True)
    model.fit(y,X)
    print(mse(y,model.predict(X)))