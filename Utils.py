def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    T.backends.cudnn.deterministic = True

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), keepdims=True)
