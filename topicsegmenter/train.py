import os
import torch
import inspect
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, get_linear_schedule_with_warmup
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm

from src.config import Config
from src.dataset import SegmentationDataset, SyntheticDataset
from src.utils import set_seed

def _model_accepts_token_type_ids(model) -> bool:
    try:
        return "token_type_ids" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False

def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom

def train_epoch(model, loader, optimizer, scheduler, cfg, writer, epoch, global_step, use_token_type_ids: bool):
    model.train()
    total_loss = 0
    
    loop = tqdm(loader, desc=f"Train Epoch {epoch+1}", leave=False)
    for batch in loop:
        optimizer.zero_grad()
        
        if cfg.architecture == "bi_encoder":
            # Bi-Encoder: Embed left and right separately
            input_ids_l = batch['input_ids_left'].to(cfg.device)
            mask_l = batch['attention_mask_left'].to(cfg.device)
            input_ids_r = batch['input_ids_right'].to(cfg.device)
            mask_r = batch['attention_mask_right'].to(cfg.device)
            labels = batch['labels'].to(cfg.device) # 0 or 1
            
            # Forward
            out_l = model(input_ids=input_ids_l, attention_mask=mask_l)
            out_r = model(input_ids=input_ids_r, attention_mask=mask_r)
            
            # EmbeddingGemma is a sentence-embedding model; use mean pooling + L2 normalize (Sentence-Transformers default)
            emb_l = F.normalize(_mean_pool(out_l.last_hidden_state, mask_l), p=2, dim=1)
            emb_r = F.normalize(_mean_pool(out_r.last_hidden_state, mask_r), p=2, dim=1)
            
            # Loss: CosineEmbeddingLoss expects 1 for similar, -1 for dissimilar
            # Our labels: 0 for same segment (similar), 1 for split (dissimilar)
            target = 1.0 - 2.0 * labels.float() # 0 -> 1, 1 -> -1
            
            criterion = torch.nn.CosineEmbeddingLoss(margin=0.3)
            loss = criterion(emb_l, emb_r, target)
        else:
            # Cross-Encoder: Paired input
            input_ids = batch['input_ids'].to(cfg.device)
            mask = batch['attention_mask'].to(cfg.device)
            token_type = batch['token_type_ids'].to(cfg.device) if use_token_type_ids else None
            labels = batch['labels'].to(cfg.device)

            model_inputs = {"input_ids": input_ids, "attention_mask": mask}
            if use_token_type_ids:
                model_inputs["token_type_ids"] = token_type
            outputs = model(**model_inputs)
            
            # Label Smoothing Loss
            criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = criterion(outputs.logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        
        # Logging step
        if global_step % 10 == 0:
            writer.add_scalar('Train/Loss_Step', loss.item(), global_step)
            writer.add_scalar('Train/LR', scheduler.get_last_lr()[0], global_step)
        
        loop.set_postfix(loss=loss.item())
        global_step += 1

    avg_loss = total_loss / len(loader)
    writer.add_scalar('Train/Loss_Epoch', avg_loss, epoch)
    return global_step

def evaluate(model, loader, cfg, writer, epoch, use_token_type_ids: bool):
    model.eval()
    all_preds = []
    all_labels = []
    all_sims = []
    total_loss = 0

    loop = tqdm(loader, desc=f"Val Epoch {epoch+1}", leave=False)
    with torch.no_grad():
        for batch in loop:
            if cfg.architecture == "bi_encoder":
                input_ids_l = batch['input_ids_left'].to(cfg.device)
                mask_l = batch['attention_mask_left'].to(cfg.device)
                input_ids_r = batch['input_ids_right'].to(cfg.device)
                mask_r = batch['attention_mask_right'].to(cfg.device)
                labels = batch['labels'].to(cfg.device)
                
                out_l = model(input_ids=input_ids_l, attention_mask=mask_l)
                out_r = model(input_ids=input_ids_r, attention_mask=mask_r)
                
                emb_l = F.normalize(_mean_pool(out_l.last_hidden_state, mask_l), p=2, dim=1)
                emb_r = F.normalize(_mean_pool(out_r.last_hidden_state, mask_r), p=2, dim=1)
                
                # Similarity
                sim = F.cosine_similarity(emb_l, emb_r)
                all_sims.extend(sim.detach().cpu().numpy())
                
                target = 1.0 - 2.0 * labels.float()
                criterion = torch.nn.CosineEmbeddingLoss(margin=0.3)
                loss = criterion(emb_l, emb_r, target)
                total_loss += loss.item()
            else:
                input_ids = batch['input_ids'].to(cfg.device)
                mask = batch['attention_mask'].to(cfg.device)
                token_type = batch['token_type_ids'].to(cfg.device) if use_token_type_ids else None
                labels = batch['labels'].to(cfg.device)

                model_inputs = {"input_ids": input_ids, "attention_mask": mask}
                if use_token_type_ids:
                    model_inputs["token_type_ids"] = token_type
                outputs = model(**model_inputs)
                
                # Label Smoothing Loss (consistency with training)
                criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
                loss = criterion(outputs.logits, labels)
                total_loss += loss.item()
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_labels.extend(labels.cpu().numpy())

            if cfg.architecture != "bi_encoder":
                all_preds.extend(preds)

    # Metrics
    if cfg.architecture == "bi_encoder":
        y_true = np.asarray(all_labels, dtype=np.int64)
        sims = np.asarray(all_sims, dtype=np.float32)
        best = {"f1": -1.0, "thr": 0.5, "p": 0.0, "r": 0.0}
        for thr in np.linspace(0.30, 0.95, 66):
            y_pred = (sims < thr).astype(np.int64)
            p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
            if f1 > best["f1"]:
                best = {"f1": float(f1), "thr": float(thr), "p": float(p), "r": float(r)}
        all_preds = (sims < best["thr"]).astype(np.int64)
        precision, recall, f1 = best["p"], best["r"], best["f1"]
        acc = accuracy_score(y_true, all_preds)
        writer.add_scalar("Val/BestThr", best["thr"], epoch)
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
        acc = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(loader)

    if cfg.architecture == "bi_encoder":
        if sims.size:
            sim_min = float(sims.min())
            sim_med = float(np.percentile(sims, 50))
            sim_max = float(sims.max())
            pred_pos_rate = float(all_preds.mean())
            print(
                f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | F1: {f1:.4f} | P: {precision:.4f} | R: {recall:.4f} "
                f"| Thr(sim): {best['thr']:.3f} | Sim[min/med/max]: {sim_min:.3f}/{sim_med:.3f}/{sim_max:.3f} "
                f"| PredPos: {pred_pos_rate:.3f}"
            )
        else:
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | F1: {f1:.4f} | P: {precision:.4f} | R: {recall:.4f} | Thr(sim): {best['thr']:.3f}")
    else:
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | F1: {f1:.4f} | P: {precision:.4f} | R: {recall:.4f}")
    
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    writer.add_scalar('Val/F1', f1, epoch)
    writer.add_scalar('Val/Precision', precision, epoch)
    writer.add_scalar('Val/Recall', recall, epoch)
    
    return f1

def main():
    cfg = Config()
    set_seed(cfg.seed)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    print(f"Initializing pipeline. Architecture: {cfg.architecture}, Device: {cfg.device}")

    # 1. Tokenizer & Data
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.sep_token
    
    # 1.1 Load Synthetic Data (if available)
    synthetic_train = None
    synthetic_val = None
    if os.path.exists("dataset.json"):
        print("Found dataset.json, loading synthetic data...")
        synthetic_train = SyntheticDataset("dataset.json", tokenizer, cfg, is_train=True, epoch_multiplier=10)
        synthetic_val = SyntheticDataset("dataset.json", tokenizer, cfg, is_train=False, epoch_multiplier=2) # Меньше множитель для валидации

    # 1.2 Load Main Data
    if os.path.exists(cfg.val_path):
        train_ds = SegmentationDataset(cfg.train_path, tokenizer, cfg, is_train=True)
        val_ds = SegmentationDataset(cfg.val_path, tokenizer, cfg, is_train=False)
    else:
        # Если есть только train, сплитим его
        data = SegmentationDataset._load_jsonl(cfg.train_path)
        train_full = SegmentationDataset(cfg.train_path, tokenizer, cfg, is_train=True, data=data)
        val_full = SegmentationDataset(cfg.train_path, tokenizer, cfg, is_train=False, data=data)

        generator = torch.Generator().manual_seed(cfg.seed)
        indices = torch.randperm(len(data), generator=generator).tolist()
        train_size = int(0.9 * len(indices))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_ds = Subset(train_full, train_indices)
        val_ds = Subset(val_full, val_indices)
        
    # 1.3 Combine
    if synthetic_train:
        print(f"Combining {len(train_ds)} real train samples with {len(synthetic_train)} synthetic train samples.")
        train_ds = ConcatDataset([train_ds, synthetic_train])
    
    if synthetic_val:
        print(f"Combining {len(val_ds)} real val samples with {len(synthetic_val)} synthetic val samples.")
        val_ds = ConcatDataset([val_ds, synthetic_val])
        
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    # 2. Model setup
    if cfg.architecture == "bi_encoder":
        model = AutoModel.from_pretrained(cfg.model_name, trust_remote_code=True)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=2, trust_remote_code=True)
    
    model.to(cfg.device)
    
    if cfg.freeze_base:
        print("Freezing base model parameters (except last 2 layers)...")
        
        # Freeze all
        for param in model.parameters():
            param.requires_grad = False
            
        # Try to find layers in various common places
        layers = None
        if hasattr(model, "layers"):
            layers = model.layers
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "bert") and hasattr(model.bert, "encoder"):
            layers = model.bert.encoder.layer
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            layers = model.transformer.h
        
        # Unfreeze last 2 layers if found
        if layers is not None and len(layers) >= 2:
            print(f"Unfreezing last 2 layers (total layers: {len(layers)})")
            for layer in layers[-2:]:
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            print("Warning: Could not find layers to unfreeze. Unfreezing all parameters as fallback.")
            for param in model.parameters():
                param.requires_grad = True
        
        # Always unfreeze the head if it exists (for Cross-Encoder)
        for head_attr in ["classifier", "score"]:
            if hasattr(model, head_attr):
                print(f"Unfreezing {head_attr} head")
                for param in getattr(model, head_attr).parameters():
                    param.requires_grad = True

    use_token_type_ids = _model_accepts_token_type_ids(model)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)
    
    writer = SummaryWriter(log_dir=cfg.log_dir)
    global_step = 0
    best_f1 = 0.0

    # 3. Training Loop
    for epoch in range(cfg.epochs):
        global_step = train_epoch(model, train_loader, optimizer, scheduler, cfg, writer, epoch, global_step, use_token_type_ids)
        f1 = evaluate(model, val_loader, cfg, writer, epoch, use_token_type_ids)
        
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(cfg.save_dir)
            tokenizer.save_pretrained(cfg.save_dir)
            print(f"  >>> Best model saved with F1: {best_f1:.4f}")

    writer.close()

if __name__ == "__main__":
    main()
