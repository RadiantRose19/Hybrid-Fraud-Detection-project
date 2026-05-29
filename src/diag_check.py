import numpy as np
import os
p=os.path.dirname(__file__)
files=['results_true.npy','results_pred.npy','results_prob.npy']
for f in files:
    fp=os.path.join(p,f)
    a=np.load(fp)
    print(f, a.shape, a.dtype, 'min', float(a.min()), 'max', float(a.max()))
    print('first10', a[:10])
