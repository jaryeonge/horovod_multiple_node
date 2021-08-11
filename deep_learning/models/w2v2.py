import os
import time
from functools import partial

import horovod.torch as hvd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from deep_learning.config import W2VCfg
from deep_learning.utils.data_utils import SftpDataset
from deep_learning.utils.dl_logger import default_logger
from deep_learning.utils.etc import convert_seconds, compose_hangul

logger = default_logger()


def wav2vev2ft_collate(batch, processor):
    input_values = []
    labels = []
    for data in batch:
        input_values.append(data['input_values'])
        labels.append(data['labels'])

    try:
        input_values = processor.feature_extractor(input_values, padding=True, sampling_rate=16000, return_tensors='pt').input_values
        labels = processor.tokenizer(labels, padding=True, return_tensors='pt').input_ids
    except Exception as e:
        logger.error(e)
        logger.info(input_values)
        logger.info(labels)
        return None, None

    return input_values, labels


def train():
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    logger.info(f'Server is loaded, Rank: {hvd.rank()}')

    config = W2VCfg()

    dataset = SftpDataset(config)
    logger.info(f'{hvd.rank()}: Data is loaded.')

    processor = Wav2Vec2Processor.from_pretrained(config.processor_path)
    sampler = DistributedSampler(dataset=dataset,
                                 num_replicas=hvd.size(),
                                 rank=hvd.rank())

    dataloader = DataLoader(dataset=dataset,
                            batch_size=config.batch_size,
                            sampler=sampler,
                            collate_fn=partial(wav2vev2ft_collate, processor=processor))

    logger.info(f'{hvd.rank()}: Data loader is built.')

    model = Wav2Vec2ForCTC.from_pretrained('./w2v2_pretrained')
    for params in model.wav2vec2.parameters():
        params.requires_grad = False

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
        for input_values, labels in dataloader:
            if input_values is None:
                continue
            optimizer.zero_grad()
            input_values = input_values.cuda()
            labels = labels.cuda()

            output = model(input_values=input_values, return_dict=True, labels=labels)
            loss = output.loss

            batch_loss += loss.item()
            batch_count += 1
            batch_mean_loss = batch_loss / batch_count
            if batch_count % config.log_period == 0:
                logger.info(f'{hvd.rank()}: batch_count: {batch_count}, batch_mean_loss: {batch_mean_loss}')
                result = torch.argmax(output.logits, dim=-1)
                d = processor.batch_decode(result)[0]
                try:
                    logger.info(compose_hangul(d))
                except Exception as e:
                    logger.error(e)
                    logger.info(d)

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
