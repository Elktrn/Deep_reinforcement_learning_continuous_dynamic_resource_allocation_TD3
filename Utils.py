# def set_seed(seed):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     T.manual_seed(seed)
#     T.cuda.manual_seed(seed)
#     T.backends.cudnn.deterministic = True

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), keepdims=True)

def soft_update(target, source, tau):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(param.data)
