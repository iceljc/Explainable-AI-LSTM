import torch

params = {}

### general parameters
params['USE_CUDA'] = torch.cuda.is_available()
params['DEVICE'] = torch.device('cuda:0')

params['BATCH_SIZE'] = 32

params['DROPOUT_PROB'] = 0.3
params['EMBEDDING_DIM'] = 300
params['HIDDEN_SIZE'] = 256
params['NUM_LAYERS'] = 1

params['NUM_EPOCHS'] = 100
params['LEARNING_RATE'] = 0.0001