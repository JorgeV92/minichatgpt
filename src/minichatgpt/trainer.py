from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split

from .config import GPTConfig, TrainConfig
from .utils import save_checkpoint


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: Dataset,
        train_config: TrainConfig,
        out_dir: str | Path,
        collate_fn=None,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.config = train_config
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.collate_fn = collate_fn
        self.device = torch.device(train_config.device)
        self.model.to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )

    def fit(self) -> Path:
        dataset_size = len(self.train_dataset)
        val_size = max(1, int(0.1 * dataset_size)) if dataset_size >= 10 else 0
        train_size = dataset_size - val_size
        if val_size > 0:
            train_ds, val_ds = random_split(self.train_dataset, [train_size, val_size])
        else:
            train_ds, val_ds = self.train_dataset, None

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
        )
        val_loader = None
        if val_ds is not None:
            val_loader = DataLoader(
                val_ds,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                collate_fn=self.collate_fn,
            )

        step = 0
        best_val = float("inf")
        best_path = self.out_dir / "best_model.pt"

        for epoch in range(self.config.epochs):
            self.model.train()
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                _, loss = self.model(x, y)
                assert loss is not None
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
                step += 1

                if step % self.config.log_interval == 0:
                    print(f"[train] epoch={epoch+1} step={step} loss={loss.item():.4f}")

                if val_loader is not None and step % self.config.eval_interval == 0:
                    val_loss = self.evaluate(val_loader)
                    print(f"[eval] step={step} val_loss={val_loss:.4f}")
                    if val_loss < best_val:
                        best_val = val_loss
                        save_checkpoint(
                            best_path,
                            self.model,
                            self.optimizer,
                            {
                                "gpt_config": asdict(self.model.config),
                                "train_config": asdict(self.config),
                            },
                            step,
                        )

        if not best_path.exists():
            save_checkpoint(
                best_path,
                self.model,
                self.optimizer,
                {
                    "gpt_config": asdict(self.model.config),
                    "train_config": asdict(self.config),
                },
                step,
            )
        return best_path

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        losses: list[float] = []
        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            _, loss = self.model(x, y)
            assert loss is not None
            losses.append(loss.item())
        self.model.train()
        return sum(losses) / max(1, len(losses))