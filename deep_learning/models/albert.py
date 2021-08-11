import os
import time
from transformers import AlbertConfig, AlbertForPreTraining

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import horovod.torch as hvd

from deep_learning.config import AlbertCfg
from deep_learning.utils.data_utils import ESDataset
from deep_learning.utils.dl_logger import default_logger
from deep_learning.utils.etc import convert_seconds
from deep_learning.utils.input_utils import BatchTensor
from deep_learning.albert_tokenizer import TrainSetMaker

logger = default_logger()


def train():
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    logger.info(f'Server is loaded, Rank: {hvd.rank()}')

    config = AlbertCfg()

    preprocessor = TrainSetMaker(config)
    dataset = ESDataset(config)
    logger.info(f'{hvd.rank()}: Data is loaded.')

    dataloader = DataLoader(dataset=dataset,
                            batch_size=config.batch_size,
                            collate_fn=preprocessor.collate_fn_sop)

    logger.info(f'{hvd.rank()}: Data loader is built.')

    model_config = AlbertConfig(**config.model_param)
    model = AlbertForPreTraining(model_config)
    optimizer = optim.Adam(params=model.parameters(), lr=config.learning_rate)
    optimizer = hvd.DistributedOptimizer(
        optimizer,
        named_parameters=model.named_parameters()
    )
    model = model.cuda()

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    logger.info(f'{hvd.rank()}: Start training.')

    model.train()
    for epoch in range(config.epochs):
        start_time = time.time()
        batch_loss = 0
        batch_count = 0
        batch_mean_loss = 0
        for data in dataloader:
            optimizer.zero_grad()
            input_values = BatchTensor(data)
            input_values = input_values.cuda()

            output = model(**input_values)
            loss = output.loss

            loss.backward()
            optimizer.step()

            batch_loss += loss.item()
            batch_count += 1
            batch_mean_loss = batch_loss / batch_count

            if batch_count % config.log_period == 0:
                logger.info(f'{hvd.rank()}: batch_count: {batch_count}, batch_mean_loss: {batch_mean_loss}')

        end_time = time.time()
        logger.info(f'{hvd.rank()}: \nEpoch {epoch}\nSpend time: {convert_seconds(end_time-start_time)}\nLoss: {batch_mean_loss}')

        if hvd.rank() == 0:
            checkpoint_name = f'state_{str(epoch)}.pt'
            checkpoint_path = os.path.join(config.save_path, checkpoint_name)
            state = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, checkpoint_path)
            logger.info(f"{hvd.rank()}: Model's state_dict is saved in {checkpoint_path}")


if __name__ == '__main__':
    train()
