import os
import glob
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from model.model_VT_prompt_FG_CS_local import Model_FG_CS_VT_prompt_local
from datasets.transform import data_transform
from datasets.Sketchy_FG import Sketchy_Basic


from exp.options import opts
from datasets.sampler import CategorySampler,ValSampler,ValBatchSampler
import pdb

if __name__ == '__main__':
    # torch.set_num_threads(8)

    dataset_transforms = data_transform(opts)
    assert opts.FG is True 

    train_dataset = Sketchy_Basic(opts, dataset_transforms, mode='train')
    val_dataset = Sketchy_Basic(opts, dataset_transforms, mode='val')
    
    train_sampler = CategorySampler(train_dataset, opts.batch_size, shuffle=True, drop_last=True)
    train_loader = DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=opts.batch_size)

    val_sampler = ValBatchSampler(ValSampler(val_dataset), opts.batch_size, drop_last=False)
    val_loader = DataLoader(dataset=val_dataset, batch_sampler=val_sampler) 

    logger = TensorBoardLogger('tb_logs', name=opts.exp_name)

    monitor_metrics = 'P_1'
    
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metrics,
        dirpath='saved_models/%s'%opts.exp_name,
        filename="{epoch:02d}-{top10:.2f}",
        mode='max',
        save_last=True)

    ckpt_path = os.path.join('saved_models', opts.exp_name, opts.model_name) ## 'last.ckpt'
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    else:
        print ('resuming training or test from %s'%ckpt_path)

    trainer = Trainer(
        accelerator='gpu', devices=-1, #gpus=-1,
        min_epochs=1, max_epochs=opts.epoches,
        benchmark=True,
        logger=logger,
        log_every_n_steps=10,\
        enable_progress_bar=False, 
        check_val_every_n_epoch=opts.check_val_every_n_epoch,
        resume_from_checkpoint=ckpt_path,
        callbacks=[checkpoint_callback]
    )

    if ckpt_path is None:
        model = Model_FG_CS_VT_prompt_local()
    else:
        print ('resuming training from %s'%ckpt_path)
        model = Model_FG_CS_VT_prompt_local().load_from_checkpoint(ckpt_path, strict=False)

    if 'train' in opts.mode:
        print ('beginning training...good luck...')
        trainer.fit(model, train_loader, val_loader)

    if 'test' in opts.mode:
        print('begin test...')

        trainer.test(model.eval(), val_loader)

