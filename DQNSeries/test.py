import numpy as np
import torch
dump = np.random.rand(3,2)
print(dump)
batch_index = np.random.randint(0,2,3)
print(batch_index)
dump = torch.as_tensor(dump,dtype=torch.float32)
batch_index = torch.as_tensor(batch_index,dtype=torch.int64).unsqueeze(1)
print(dump.gather(0,batch_index))#取列了，真是个傻逼！！
print(dump.gather(1,batch_index))