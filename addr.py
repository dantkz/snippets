import numpy as np

num_pos = 2
d = 3
d1 = d+1

A = np.arange(d1*d1, dtype=np.float32).reshape([d1,d1])
idx = np.tile(np.expand_dims(np.arange(d1, dtype=np.int32),1), [num_pos,1]).reshape([num_pos,d1])
idx[1,:] += 1
result = A[:,idx[:,:-1]].transpose([1,0,2])
result = np.reshape(result,[-1])
print(result)
print("------------------")

#print(d1*(result//d1))
print(result % d1)


tmp = np.expand_dims(np.arange(d1, dtype=np.int32)*d1, 1)
tmp = np.tile(tmp, [num_pos, d])
base_canonical = np.reshape(tmp, [-1])

idx00 = idx[:,:-1]
print(idx00.shape)
idx00 = np.tile(idx00, [1,d1])
idx00 = np.reshape(idx00, [-1])
print(idx00.astype(np.float32))
#idx00 = np.reshape(np.tile(np.expand_dims(idx00, 1), [d1, 1]), [-1])
idx1 = base_canonical + idx00
tmp = (np.take(np.reshape(A, [-1]), idx1))
#print(base_canonical.astype(np.float32))
print(result - tmp%d1)
#print(idx00.astype(np.float32))

print(np.all(result - tmp==0))
