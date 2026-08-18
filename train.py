import os
from pprint import pprint
from configs.config import parser
from dataset.longitudinal_data_module import DataModule
from lightning_tools.callbacks import add_callbacks
from lightning.pytorch import seed_everything
from lightning.pytorch.strategies import DeepSpeedStrategy
import lightning.pytorch as pl
import torch
import shutil
import importlib

torch.set_float32_matmul_precision('medium')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train(args):
    callbacks = add_callbacks(args)

    trainer = pl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=DeepSpeedStrategy(config=args.deepspeed_config),
        accelerator=args.accelerator,
        precision=args.precision,
        val_check_interval = args.val_check_interval,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches = args.limit_val_batches,
        limit_test_batches = args.limit_test_batches,
        max_epochs = args.max_epochs,
        num_sanity_val_steps = args.num_sanity_val_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks["callbacks"], 
        logger=callbacks["loggers"]
    )

    dm = DataModule(args)

    model_module = importlib.import_module(f"models.model_{args.stage}")
    model_cls = model_module.LongitudinalR2GenGPT

    model = model_cls.load_from_checkpoint(args.ckpt_file, strict=False, args=args) if args.ckpt_file is not None else model_cls(args)

    if args.test:
        trainer.test(model, datamodule=dm)
    elif args.validate:
        trainer.validate(model, datamodule=dm)
    else:
        model_file = os.path.join('models', f'model_{args.stage}.py')
        shutil.copy(model_file, args.savedmodel_path)
        trainer.fit(model, datamodule=dm)

def main():
    args = parser.parse_args()
    os.makedirs(args.savedmodel_path, exist_ok=True)
    pprint(vars(args))
    seed_everything(42, workers=True)
    train(args)


if __name__ == '__main__':
    main()