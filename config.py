from yacs.config import CfgNode as CN

_C = CN()


_C.num_epochs = 100
_C.train_batch_size = 64
_C.val_batch_size = 64

_C.log_time = 10

_C.device = 'cuda'
_C.root_dir = '/netscratch/deshmukh/train_valid_test_splits/'
_C.detections_file = '/ds/images/coco-detections/coco_detections.hdf5'
_C.model_path = '/netscratch/deshmukh/thesis/GAN-Captioning/saved_models/'
_C.save_dir = 'output_logs'

_C.solver = CN()
_C.solver.lr = 5e-5
_C.solver.weight_decay = 1e-2
_C.solver.betas = (0.9, 0.999)
_C.solver.grad_clip = 1.0

_C.scheduler = CN()
_C.scheduler.warmup_steps = 1000
_C.scheduler.max_steps = 100000

_C.loss = CN()
_C.loss.balance_weight = 0.5
_C.loss.label_smoothing = 0.1