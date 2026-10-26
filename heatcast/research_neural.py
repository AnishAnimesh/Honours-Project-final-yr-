"""Matched LSTM and linear baselines for the long-history benchmark."""
import copy
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader,TensorDataset
from .models import LSTM
from .evaluation import choose_threshold


def neural_comparison(raw,anomaly,idx,lead,y,train,val,test,lat,lon,epochs=20):
    torch.set_num_threads(4);torch.manual_seed(42)
    # Fitted on observations no later than the latest training target; all precede validation.
    end=int(idx[train].max())
    values=np.concatenate([raw,anomaly[:,:,None]],axis=-1)
    mean=values[:end+1].mean(axis=(0,1));std=np.maximum(values[:end+1].std(axis=(0,1)),1e-6)
    values=((values-mean)/std).astype('float32')
    issue=idx-lead
    seq=np.stack([values[issue-k] for k in range(13,-1,-1)],axis=2).reshape(-1,14,values.shape[-1])
    loaders={}
    for name,selection in [('train',train),('validation',val),('test',test)]:
        rows=(selection[:,None]*len(lat)+np.arange(len(lat))[None]).ravel()
        ds=TensorDataset(torch.from_numpy(seq[rows].copy()),torch.from_numpy(y[selection].ravel().astype('float32')))
        loaders[name]=DataLoader(ds,batch_size=256,shuffle=name=='train')
    model=LSTM(values.shape[-1]);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    rate=y[train].mean();weight=torch.tensor(np.sqrt((1-rate)/max(rate,1e-6)))
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    best=float('inf');state=None;stale=0;history=[]
    for epoch in range(epochs):
        model.train();losses=[]
        for xb,yb in loaders['train']:
            optimizer.zero_grad();loss=criterion(model(xb),yb);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(loss.item())
        model.eval()
        with torch.no_grad():v=sum(criterion(model(xb),yb).item()*len(xb) for xb,yb in loaders['validation'])/len(loaders['validation'].dataset)
        history.append(dict(epoch=epoch+1,training_loss=float(np.mean(losses)),validation_loss=v))
        if v<best:best=v;state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if stale>=4:break
    model.load_state_dict(state);model.eval();pred={}
    with torch.no_grad():
        for name in ['validation','test']:
            pred[name]=np.concatenate([model(xb).sigmoid().numpy() for xb,_ in loaders[name]]).reshape(-1,len(lat))
    cutoff=choose_threshold(y[val],pred['validation'])
    return dict(state_dict=model.state_dict(),mean=mean.tolist(),std=std.tolist(),history=history,threshold=cutoff,input_features=7),pred


def linear_comparison(x,y,train,val,test):
    model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=500,random_state=42))
    model.fit(x[train].reshape(-1,x.shape[-1]),y[train].ravel())
    pv=model.predict_proba(x[val].reshape(-1,x.shape[-1]))[:,1].reshape(len(val),-1)
    pt=model.predict_proba(x[test].reshape(-1,x.shape[-1]))[:,1].reshape(len(test),-1)
    return model,pv,pt,choose_threshold(y[val],pv)
