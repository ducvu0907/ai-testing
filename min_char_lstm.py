import numpy as np
from random import uniform

# based on Karpathy's min-char-rnn
data = open("input.txt", 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

hidden_size = 100
block_size = 25
learning_rate = 1e-1

Wf = np.random.randn(hidden_size, hidden_size + vocab_size) * 0.01 # forget gate
Wi = np.random.randn(hidden_size, hidden_size + vocab_size) * 0.01 # input gate
Wo = np.random.randn(hidden_size, hidden_size + vocab_size) * 0.01 # output gate
Wcd = np.random.randn(hidden_size, hidden_size + vocab_size) * 0.01 # candidate

bf = np.zeros((hidden_size, 1))
bo = np.zeros((hidden_size, 1))
bi = np.zeros((hidden_size, 1))
bcd = np.zeros((hidden_size, 1))

# output layer
Wy = np.random.randn(vocab_size, hidden_size) * 0.01
by = np.zeros((vocab_size, 1))

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def lossfn(inputs, targets, hprev, cprev):
  xs, hs, hxs, ys, ps, fgs, igs, ogs, cds, cs = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
  # prev state and cell
  hs[-1] = np.copy(hprev)
  cs[-1] = np.copy(cprev)
  loss = 0

  # forward pass
  for t in range(len(inputs)):
    xs[t] = np.zeros((vocab_size,1))
    xs[t][inputs[t]] = 1
    hxs[t] = np.concatenate((hs[t-1], xs[t]), axis=0)
    fgs[t] = sigmoid(np.dot(Wf, hxs[t]) + bf)
    igs[t] = sigmoid(np.dot(Wi, hxs[t]) + bi)
    cds[t] = np.tanh(np.dot(Wcd, hxs[t]) + bcd)
    ogs[t] = sigmoid(np.dot(Wo, hxs[t]) + bo) 
    # this timestep cell and state
    cs[t]= fgs[t] * cs[t-1] + igs[t] * cds[t]
    hs[t] = ogs[t] * np.tanh(cs[t])

    ys[t] = np.dot(Wy, hs[t]) + by
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
    loss += -np.log(ps[t][targets[t]])

  # grads
  dWf = np.zeros_like(Wf)
  dbf = np.zeros_like(bf)
  dWi = np.zeros_like(Wi)
  dbi = np.zeros_like(bi)
  dWcd = np.zeros_like(Wcd)
  dbcd = np.zeros_like(bcd)
  dWo = np.zeros_like(Wo)
  dbo = np.zeros_like(bo)
  dWy = np.zeros_like(Wy)
  dby = np.zeros_like(by)

  dhnext = np.zeros_like(hs[-1])
  dcnext = np.zeros_like(cs[-1])

  # backward pass
  for t in reversed(range(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1
    dWy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Wy.T, dy) + dhnext # through time
    do = dh * np.tanh(cs[t]) # output
    dca = dh * ogs[t]
    dc = (1 - cs[t] ** 2) * dca + dcnext # through tanh and time
    df = dc * cs[t-1]
    di = dc * cds[t]
    dcd = dc * igs[t]
    do_raw = ogs[t] * (1 - ogs[t]) * do # backprop through sigmoid
    dbo += do_raw
    dWo += np.dot(do_raw, hxs[t].T)
    dcd_raw = (1 - cds[t] ** 2) * dcd # through tanh
    dbcd += dcd_raw
    dWcd += np.dot(dcd_raw, hxs[t].T)
    di_raw = igs[t] * (1 - igs[t]) * di # through sigmoid
    dbi += di_raw
    dWi += np.dot(di_raw, hxs[t].T)
    df_raw = fgs[t] * (1 - fgs[t]) * df # through sigmoid
    dbf += df_raw
    dWf += np.dot(df_raw, hxs[t].T)

    dhxf = np.dot(Wf.T, df_raw)
    dhxi = np.dot(Wi.T, di_raw)
    dhxo = np.dot(Wo.T, df_raw)
    dhxcd = np.dot(Wcd.T, dcd_raw)
    dhx = dhxf + dhxi + dhxo + dhxcd

    dhnext = dhx[:hidden_size, :]
    dcnext = fgs[t] * dc

  # clip gradient
  for dparam in [dWf, dWi, dWcd, dWo, dWy, dbf, dbi, dbo, dby]:
    np.clip(dparam, -5, 5, out=dparam)

  return loss, dWf, dWi, dWcd, dWo, dWy, dbf, dbi, dbo, dby, hs[len(inputs) - 1], cs[len(inputs) - 1]

def sample(h, c, seed_ix, n):
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for _ in range(n):
    hx = np.concatenate((h, x), axis=0)
    f = sigmoid(np.dot(Wf, hx) + bf)
    i = sigmoid(np.dot(Wi, hx) + bi)
    cd = np.tanh(np.dot(Wcd, hx) + bcd)
    o = sigmoid(np.dot(Wo, hx) + bo)
    c = f * c + i * cd
    h = o * np.tanh(c)
    y = np.dot(Wy, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ''.join(ix_to_char[ix] for ix in ixes)

def grad_check(inputs, targets, hprev, cprev):
  global Wf, Wi, bf, bi, Wcd, bcd, Wo, bo, Wy, by
  num_checks, delta = 10, 1e-5
  _, dWf, dWi, dWcd, dWo, dWy, dbf, dbi, dbo, dby, _, _ = lossfn(inputs, targets, hprev, cprev)
  for param, dparam, name in zip([Wf, Wi, Wcd, Wo, Wy, bf, bi, bo, by], 
                                [dWf, dWi, dWcd, dWo, dWy, dbf, dbi, dbo, dby], 
                                ["Wf", "Wi", "Wcd", "Wo", "Wy", "bf", "bi", "bo", "by"]):
    assert param.shape == dparam.shape, "dims don't match"
    print(f"checking gradients for {name}")
    for _ in range(num_checks):
      ri = int(uniform(0, param.size)) # sample from uniform distribution
      # evaluate cost at (x + delta) and (x - delta)
      old_val = param.flat[ri]
      param.flat[ri] = old_val + delta
      cg0 = lossfn(inputs, targets, hprev, cprev)[0] # loss for x + delta
      param.flat[ri] = old_val - delta
      cg1 = lossfn(inputs, targets, hprev, cprev)[0] # loss for x - delta
      param.flat[ri] = old_val # reset old value for this parameter
      # fetch both numerical and analytic gradient
      grad_analytic = dparam.flat[ri]
      grad_numerical = (cg0 - cg1) / ( 2 * delta )
      rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
      print(f"{grad_numerical}, {grad_analytic} => {rel_error}")

# simple grad check wrapper
def init_grad_check():
  inputs = [char_to_ix[ch] for ch in data[:block_size]]
  targets = [char_to_ix[ch] for ch in data[1:block_size+1]]
  hprev = np.zeros((hidden_size, 1))
  cprev = np.zeros((hidden_size, 1))
  grad_check(inputs, targets, hprev, cprev)


if __name__ == "__main__":
  init_grad_check()

  n, p = 0, 0
  mWf = np.zeros_like(Wf)
  mbf = np.zeros_like(bf)
  mWi = np.zeros_like(Wi)
  mbi = np.zeros_like(bi)
  mWcd = np.zeros_like(Wcd)
  mbcd = np.zeros_like(bcd)
  mWo = np.zeros_like(Wo)
  mbo = np.zeros_like(bo)
  mWy = np.zeros_like(Wy)
  mby = np.zeros_like(by)
  smooth_loss = -np.log(1.0 / vocab_size) * block_size

  while True:
    if p + block_size + 1 >= len(data) or n == 0: 
      hprev = np.zeros((hidden_size, 1))
      cprev = np.zeros((hidden_size, 1))
      p = 0

    inputs = [char_to_ix[ch] for ch in data[p : p+block_size]]
    targets = [char_to_ix[ch] for ch in data[p+1 : p+block_size+1]]

    if n % 1000 == 0:
      sample_text = sample(hprev, cprev, inputs[0], 200)
      print(sample_text)

    loss, dWf, dWi, dWcd, dWo, dWy, dbf, dbi, dbo, dby, hprev, cprev = lossfn(inputs, targets, hprev, cprev)

    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % 1000 == 0:
      print(f"loss: {smooth_loss}, iter: {n}")
    
    # update parameters
    for param, dparam, mem in zip([Wf, Wi, Wcd, Wo, Wy, bf, bi, bo, by], 
                                  [dWf, dWi, dWcd, dWo, dWy, dbf, dbi, dbo, dby], 
                                  [mWf, mWi, mWcd, mWo, mWy, mbf, mbi, mbo, mby]):
      mem += dparam * dparam
      param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad

    p += block_size # move data pointer
    n += 1 # iteration counter
