"""
Phase 1 Training Script - Train Transformer Variants from Scratch

This module provides complete training pipelines for:
1. TextClassifier (Encoder-Only): Abstract classification
2. DecoderOnlyLM: Next-token prediction on abstracts
3. EncoderDecoderTransformer: Abstract summarization (TL;DR)

Features:
- Custom BPE tokenizer training
- TensorBoard logging
- Checkpointing
- Evaluation metrics (Accuracy, Perplexity, ROUGE)
- Qualitative output examples

References:
- The Annotated Transformer (Harvard NLP)
- minGPT (Andrej Karpathy)
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

# Tokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

# Evaluation
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge_score not installed. ROUGE evaluation disabled.")

# TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False
    print("Warning: TensorBoard not installed. Logging disabled.")

# Local imports
from .models import TextClassifier, DecoderOnlyLM, EncoderDecoderTransformer, count_parameters

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
TOKENIZER_DIR = MODELS_DIR / "tokenizer"
CHECKPOINTS_DIR = MODELS_DIR / "phase1_checkpoints"

# Ensure directories exist
for d in [TOKENIZER_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# TOKENIZER
# =============================================================================

def train_bpe_tokenizer(
    texts: List[str],
    vocab_size: int = 10000,
    save_path: Path = None
) -> Tokenizer:
    """
    Train a BPE tokenizer on the given texts.
    
    Args:
        texts: List of text strings to train on
        vocab_size: Vocabulary size
        save_path: Path to save tokenizer
    
    Returns:
        Trained tokenizer
    """
    save_path = save_path or TOKENIZER_DIR / "tokenizer.json"
    
    # Initialize BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    # Train
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]"]
    )
    
    tokenizer.train_from_iterator(texts, trainer)
    
    # Add post-processing for BOS/EOS
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [SEP] $B:1 [EOS]:1",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ]
    )
    
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")
    tokenizer.enable_truncation(max_length=256)
    
    # Save
    tokenizer.save(str(save_path))
    print(f"✅ Tokenizer saved to {save_path}")
    
    return tokenizer


def load_tokenizer(path: Path = None) -> Tokenizer:
    """Load tokenizer from file."""
    path = path or TOKENIZER_DIR / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {path}. Train one first.")
    return Tokenizer.from_file(str(path))


# =============================================================================
# DATASETS
# =============================================================================

class AbstractClassificationDataset(Dataset):
    """Dataset for abstract classification (Encoder-Only model)."""
    
    def __init__(
        self,
        papers: List[Dict],
        tokenizer: Tokenizer,
        label2id: Dict[str, int],
        max_len: int = 256
    ):
        self.papers = papers
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len
        self.pad_token_id = tokenizer.token_to_id("[PAD]")
        
        # Update tokenizer truncation
        self.tokenizer.enable_truncation(max_length=max_len)
    
    def __len__(self):
        return len(self.papers)
    
    def __getitem__(self, idx):
        paper = self.papers[idx]
        text = paper['abstract']
        label = self.label2id.get(paper.get('topic', 'Other'), 0)
        
        # Tokenize
        encoded = self.tokenizer.encode(text)
        ids = encoded.ids
        
        # Pad to max_len
        if len(ids) < self.max_len:
            padding_length = self.max_len - len(ids)
            ids = ids + [self.pad_token_id] * padding_length
            attention_mask = [1.0] * len(encoded.ids) + [0.0] * padding_length
        else:
            ids = ids[:self.max_len]
            attention_mask = [1.0] * self.max_len
        
        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.float)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }


class LanguageModelDataset(Dataset):
    """Dataset for language modeling (Decoder-Only model)."""
    
    def __init__(
        self,
        papers: List[Dict],
        tokenizer: Tokenizer,
        max_len: int = 256,
        stride: int = 128
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Concatenate all abstracts
        all_text = " ".join([p['abstract'] for p in papers])
        
        # Tokenize entire corpus
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        encoded = self.tokenizer.encode(all_text)
        self.all_ids = encoded.ids
        
        # Create overlapping chunks
        self.chunks = []
        for i in range(0, len(self.all_ids) - max_len, stride):
            self.chunks.append(self.all_ids[i:i + max_len])
        
        # Ensure at least one chunk
        if len(self.chunks) == 0 and len(self.all_ids) > 0:
            self.chunks.append(self.all_ids[:max_len])
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        ids = self.chunks[idx]
        pad_token_id = self.tokenizer.token_to_id("[PAD]")
        
        # Ensure exact length
        if len(ids) < self.max_len:
            ids = ids + [pad_token_id] * (self.max_len - len(ids))
        elif len(ids) > self.max_len:
            ids = ids[:self.max_len]
        
        input_ids = torch.tensor(ids, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'labels': input_ids.clone()  # For LM, labels = inputs (shifted in model)
        }


class SummarizationDataset(Dataset):
    """Dataset for summarization (Encoder-Decoder model) with proper BOS/EOS handling."""
    
    def __init__(
        self,
        papers: List[Dict],
        tokenizer: Tokenizer,
        max_src_len: int = 256,
        max_tgt_len: int = 64
    ):
        self.papers = papers
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        
        # Special token IDs
        self.bos_id = tokenizer.token_to_id("[BOS]")
        self.eos_id = tokenizer.token_to_id("[EOS]")
        self.pad_id = tokenizer.token_to_id("[PAD]")
    
    def __len__(self):
        return len(self.papers)
    
    def __getitem__(self, idx):
        paper = self.papers[idx]
        
        # Source: full abstract
        source = paper['abstract']
        
        # Target: Create TL;DR from title + first key phrases
        # Extract key information for pseudo-summary
        title = paper.get('title', '')[:80]
        sentences = source.split('. ')
        first_sent = sentences[0][:100] if sentences else ""
        target = f"TL;DR: {title}. {first_sent}"
        
        # Tokenize source (no special tokens needed for encoder)
        self.tokenizer.enable_truncation(max_length=self.max_src_len - 2)
        src_encoded = self.tokenizer.encode(source)
        src_ids = src_encoded.ids
        
        # Tokenize target
        self.tokenizer.enable_truncation(max_length=self.max_tgt_len - 2)  # Leave room for BOS/EOS
        tgt_encoded = self.tokenizer.encode(target)
        tgt_tokens = tgt_encoded.ids
        
        # For decoder input: [BOS] + target tokens (teacher forcing)
        tgt_ids = [self.bos_id] + tgt_tokens
        
        # For labels: target tokens + [EOS] (what we predict)
        labels = tgt_tokens + [self.eos_id]
        
        # Pad source
        src_padding = self.max_src_len - len(src_ids)
        src_ids = src_ids + [self.pad_id] * src_padding
        
        # Pad target input and labels to same length
        tgt_padding = self.max_tgt_len - len(tgt_ids)
        tgt_ids = tgt_ids + [self.pad_id] * tgt_padding
        labels = labels + [self.pad_id] * (self.max_tgt_len - len(labels))
        
        return {
            'src_ids': torch.tensor(src_ids[:self.max_src_len], dtype=torch.long),
            'tgt_ids': torch.tensor(tgt_ids[:self.max_tgt_len], dtype=torch.long),
            'labels': torch.tensor(labels[:self.max_tgt_len], dtype=torch.long)
        }


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_papers(data_path: Path = None) -> List[Dict]:
    """Load papers from JSONL file."""
    data_path = data_path or DATA_DIR / "abstracts.jsonl"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    papers = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                papers.append(json.loads(line))
    
    return papers


def split_data(papers: List[Dict], train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
    """Split papers into train/val sets."""
    random.shuffle(papers)
    split_idx = int(len(papers) * train_ratio)
    return papers[:split_idx], papers[split_idx:]


# =============================================================================
# TRAINER CLASSES
# =============================================================================

class LabelSmoothingLoss(nn.Module):
    """Cross entropy with label smoothing for regularization."""
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'max'
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop


def get_linear_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then linear decay."""
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class ClassifierTrainer:
    """Trainer for the Encoder-Only Text Classifier with improved regularization."""
    
    def __init__(
        self,
        model: TextClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer: Tokenizer,
        label2id: Dict[str, int],
        device: torch.device,
        lr: float = 1e-4,
        epochs: int = 10,
        log_dir: Path = None,
        label_smoothing: float = 0.1,
        patience: int = 7
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.id2label = {v: k for k, v in label2id.items()}
        self.device = device
        self.epochs = epochs
        
        # Improved optimizer with stronger weight decay
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.05)
        
        # Warmup + cosine schedule
        total_steps = epochs * len(train_loader)
        warmup_steps = int(0.1 * total_steps)
        self.scheduler = get_linear_warmup_scheduler(self.optimizer, warmup_steps, total_steps)
        self.step_scheduler_per_batch = True
        
        # Label smoothing for regularization
        self.criterion = LabelSmoothingLoss(num_classes=len(label2id), smoothing=label_smoothing)
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=patience, mode='max')
        
        # TensorBoard
        self.writer = None
        if TB_AVAILABLE and log_dir:
            self.writer = SummaryWriter(log_dir / "classifier")
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch with mixup augmentation."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Step scheduler per batch for warmup
            if self.step_scheduler_per_batch:
                self.scheduler.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        return total_loss / len(self.val_loader), correct / total
    
    def train(self) -> Dict:
        """Full training loop with early stopping."""
        best_acc = 0
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, val_acc = self.evaluate()
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            
            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'label2id': self.label2id
                }, CHECKPOINTS_DIR / "classifier_best.pt")
            
            # Early stopping check
            if self.early_stopping(val_acc):
                print(f"⚠️ Early stopping triggered at epoch {epoch+1}")
                break
        
        if self.writer:
            self.writer.close()
        
        print(f"✅ Best validation accuracy: {best_acc:.4f}")
        return history
    
    @torch.no_grad()
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict class for a single text."""
        self.model.eval()
        
        encoded = self.tokenizer.encode(text)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([encoded.attention_mask], dtype=torch.float, device=self.device)
        
        logits = self.model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=-1)
        pred_id = logits.argmax(dim=-1).item()
        confidence = probs[0, pred_id].item()
        
        return self.id2label[pred_id], confidence


class LMTrainer:
    """Trainer for the Decoder-Only Language Model with improved training."""
    
    def __init__(
        self,
        model: DecoderOnlyLM,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer: Tokenizer,
        device: torch.device,
        lr: float = 3e-4,
        epochs: int = 20,
        log_dir: Path = None,
        patience: int = 10
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.device = device
        self.epochs = epochs
        
        # Improved optimizer
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.98))
        
        # Warmup scheduler
        total_steps = epochs * len(train_loader)
        warmup_steps = int(0.1 * total_steps)
        self.scheduler = get_linear_warmup_scheduler(self.optimizer, warmup_steps, total_steps)
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=patience, mode='min')
        
        self.writer = None
        if TB_AVAILABLE and log_dir:
            self.writer = SummaryWriter(log_dir / "lm")
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            _, loss = self.model(input_ids, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'ppl': np.exp(min(loss.item(), 10))})
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        """Evaluate on validation set. Returns loss and perplexity."""
        self.model.eval()
        total_loss = 0
        
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            _, loss = self.model(input_ids, labels)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        perplexity = np.exp(min(avg_loss, 20))  # Cap to avoid overflow
        return avg_loss, perplexity
    
    def train(self) -> Dict:
        """Full training loop with early stopping."""
        best_ppl = float('inf')
        history = {'train_loss': [], 'val_loss': [], 'val_ppl': []}
        
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, val_ppl = self.evaluate()
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_ppl'].append(val_ppl)
            
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val PPL={val_ppl:.2f}")
            
            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Perplexity/val', val_ppl, epoch)
            
            # Save best model
            if val_ppl < best_ppl:
                best_ppl = val_ppl
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_ppl': val_ppl
                }, CHECKPOINTS_DIR / "lm_best.pt")
            
            # Early stopping check
            if self.early_stopping(val_loss):
                print(f"⚠️ Early stopping triggered at epoch {epoch+1}")
                break
            
            # Generate sample every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.generate_sample()
        
        if self.writer:
            self.writer.close()
        
        print(f"✅ Best validation perplexity: {best_ppl:.2f}")
        return history
    
    @torch.no_grad()
    def generate_sample(self, prompt: str = "In this paper, we"):
        """Generate sample text."""
        self.model.eval()
        
        encoded = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=self.device)
        
        generated = self.model.generate(
            input_ids,
            max_new_tokens=50,
            temperature=0.8,
            top_k=40
        )
        
        output = self.tokenizer.decode(generated[0].tolist())
        print(f"\n📝 Sample: {output}\n")
        return output


class Seq2SeqTrainer:
    """Trainer for the Encoder-Decoder Summarization Model with improved training."""
    
    def __init__(
        self,
        model: EncoderDecoderTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer: Tokenizer,
        device: torch.device,
        lr: float = 3e-4,
        epochs: int = 20,
        log_dir: Path = None,
        patience: int = 10
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.device = device
        self.epochs = epochs
        
        # Get PAD token for loss masking
        self.pad_id = tokenizer.token_to_id("[PAD]")
        
        # Use stronger weight decay and warmup
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.98))
        
        # Warmup scheduler
        total_steps = epochs * len(train_loader)
        warmup_steps = int(0.1 * total_steps)
        self.scheduler = get_linear_warmup_scheduler(self.optimizer, warmup_steps, total_steps)
        
        # Loss that ignores PAD tokens
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id, label_smoothing=0.1)
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=patience, mode='min')
        
        # ROUGE scorer
        self.rouge_scorer = None
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        self.writer = None
        if TB_AVAILABLE and log_dir:
            self.writer = SummaryWriter(log_dir / "seq2seq")
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch with proper loss computation."""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            src_ids = batch['src_ids'].to(self.device)
            tgt_ids = batch['tgt_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass - get logits
            logits, _ = self.model(src_ids, tgt_ids, labels)
            
            # Reshape for cross entropy: [batch * seq, vocab] vs [batch * seq]
            vocab_size = logits.size(-1)
            loss = self.criterion(
                logits.view(-1, vocab_size),
                labels.view(-1)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)  # Tighter clipping
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def evaluate(self) -> Tuple[float, Dict]:
        """Evaluate on validation set. Returns loss and ROUGE scores."""
        self.model.eval()
        total_loss = 0
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        num_batches = 0
        
        for batch in self.val_loader:
            src_ids = batch['src_ids'].to(self.device)
            tgt_ids = batch['tgt_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Compute loss with our criterion
            logits, _ = self.model(src_ids, tgt_ids, labels)
            vocab_size = logits.size(-1)
            loss = self.criterion(logits.view(-1, vocab_size), labels.view(-1))
            total_loss += loss.item()
            num_batches += 1
            
            # Generate summaries for ROUGE (only every few batches to save time)
            if self.rouge_scorer is not None and num_batches <= 5:
                bos_id = self.tokenizer.token_to_id("[BOS]")
                eos_id = self.tokenizer.token_to_id("[EOS]")
                generated = self.model.generate(src_ids, max_new_tokens=50, bos_token_id=bos_id, eos_token_id=eos_id)
                
                for i in range(min(generated.size(0), 4)):  # Limit samples
                    pred_text = self.tokenizer.decode(generated[i].tolist())
                    ref_text = self.tokenizer.decode(labels[i].tolist())
                    
                    # Clean up special tokens
                    for tok in ['[PAD]', '[BOS]', '[EOS]', '[UNK]']:
                        pred_text = pred_text.replace(tok, '')
                        ref_text = ref_text.replace(tok, '')
                    pred_text = pred_text.strip()
                    ref_text = ref_text.strip()
                    
                    if pred_text and ref_text:
                        scores = self.rouge_scorer.score(ref_text, pred_text)
                        for key in rouge_scores:
                            rouge_scores[key].append(scores[key].fmeasure)
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_rouge = {k: np.mean(v) if v else 0 for k, v in rouge_scores.items()}
        
        return avg_loss, avg_rouge
    
    def train(self) -> Dict:
        """Full training loop with early stopping."""
        best_loss = float('inf')
        history = {'train_loss': [], 'val_loss': [], 'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, rouge = self.evaluate()
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            for key in ['rouge1', 'rouge2', 'rougeL']:
                history[key].append(rouge.get(key, 0))
            
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            print(f"         ROUGE-1={rouge.get('rouge1', 0):.4f}, ROUGE-2={rouge.get('rouge2', 0):.4f}, ROUGE-L={rouge.get('rougeL', 0):.4f}")
            
            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                for key, val in rouge.items():
                    self.writer.add_scalar(f'ROUGE/{key}', val, epoch)
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'rouge': rouge
                }, CHECKPOINTS_DIR / "seq2seq_best.pt")
            
            # Early stopping check
            if self.early_stopping(val_loss):
                print(f"⚠️ Early stopping triggered at epoch {epoch+1}")
                break
            
            # Generate sample every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.generate_sample()
        
        if self.writer:
            self.writer.close()
        
        print(f"✅ Best validation loss: {best_loss:.4f}")
        return history
    
    @torch.no_grad()
    def generate_sample(self, text: str = None):
        """Generate a sample summary."""
        self.model.eval()
        
        if text is None:
            # Use first validation sample
            batch = next(iter(self.val_loader))
            src_ids = batch['src_ids'][:1].to(self.device)
            ref_text = self.tokenizer.decode(batch['labels'][0].tolist())
        else:
            self.tokenizer.enable_truncation(max_length=256)
            encoded = self.tokenizer.encode(text)
            src_ids = torch.tensor([encoded.ids], dtype=torch.long, device=self.device)
            ref_text = None
        
        bos_id = self.tokenizer.token_to_id("[BOS]")
        eos_id = self.tokenizer.token_to_id("[EOS]")
        
        generated = self.model.generate(src_ids, max_new_tokens=50, bos_token_id=bos_id, eos_token_id=eos_id)
        output = self.tokenizer.decode(generated[0].tolist())
        
        print(f"\n📝 Generated Summary: {output}")
        if ref_text:
            print(f"📚 Reference: {ref_text}\n")
        
        return output


# =============================================================================
# MAIN TRAINING FUNCTIONS
# =============================================================================

def train_classifier(
    papers: List[Dict] = None,
    d_model: int = 256,
    num_heads: int = 4,
    num_layers: int = 4,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-4
) -> Dict:
    """
    Train the Encoder-Only classifier.
    
    Returns training history and evaluation results.
    """
    print("\n" + "="*60)
    print("🎯 TRAINING ENCODER-ONLY CLASSIFIER")
    print("="*60)
    
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
    
    # Load data
    if papers is None:
        papers = load_papers()
    print(f"Loaded {len(papers)} papers")
    
    # Create label mapping
    topics = list(set(p.get('topic', 'Other') for p in papers))
    label2id = {t: i for i, t in enumerate(sorted(topics))}
    print(f"Classes: {label2id}")
    
    # Train tokenizer
    texts = [p['abstract'] for p in papers]
    tokenizer = train_bpe_tokenizer(texts, vocab_size=8000)
    vocab_size = tokenizer.get_vocab_size()
    
    # Split data
    train_papers, val_papers = split_data(papers.copy())
    
    # Create datasets
    train_dataset = AbstractClassificationDataset(train_papers, tokenizer, label2id)
    val_dataset = AbstractClassificationDataset(val_papers, tokenizer, label2id)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model with higher dropout for regularization
    model = TextClassifier(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=len(label2id),
        dropout=0.3  # Increased from 0.1 to reduce overfitting
    )
    print(f"Model: {count_parameters(model):,} parameters")
    
    # Train
    trainer = ClassifierTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        label2id=label2id,
        device=device,
        lr=lr,
        epochs=epochs,
        log_dir=LOGS_DIR
    )
    
    history = trainer.train()
    
    # Show sample predictions
    print("\n📊 Sample Predictions:")
    for paper in val_papers[:3]:
        pred, conf = trainer.predict(paper['abstract'])
        actual = paper.get('topic', 'Unknown')
        status = "✅" if pred == actual else "❌"
        print(f"{status} Predicted: {pred} ({conf:.2%}) | Actual: {actual}")
    
    return history


def train_language_model(
    papers: List[Dict] = None,
    d_model: int = 256,
    num_heads: int = 4,
    num_layers: int = 4,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 3e-4
) -> Dict:
    """
    Train the Decoder-Only language model.
    
    Returns training history.
    """
    print("\n" + "="*60)
    print("📝 TRAINING DECODER-ONLY LANGUAGE MODEL")
    print("="*60)
    
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
    
    # Load data
    if papers is None:
        papers = load_papers()
    print(f"Loaded {len(papers)} papers")
    
    # Load or train tokenizer
    try:
        tokenizer = load_tokenizer()
    except FileNotFoundError:
        texts = [p['abstract'] for p in papers]
        tokenizer = train_bpe_tokenizer(texts, vocab_size=8000)
    
    vocab_size = tokenizer.get_vocab_size()
    
    # Split data
    train_papers, val_papers = split_data(papers.copy())
    
    # Create datasets
    train_dataset = LanguageModelDataset(train_papers, tokenizer, max_len=128)
    val_dataset = LanguageModelDataset(val_papers, tokenizer, max_len=128)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"Train chunks: {len(train_dataset)}, Val chunks: {len(val_dataset)}")
    
    # Create model with moderate dropout
    model = DecoderOnlyLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=128,
        dropout=0.15  # Slightly increased for LM
    )
    print(f"Model: {count_parameters(model):,} parameters")
    
    # Train
    trainer = LMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        lr=lr,
        epochs=epochs,
        log_dir=LOGS_DIR
    )
    
    history = trainer.train()
    
    # Generate samples
    print("\n📝 Generated Samples:")
    prompts = ["We propose", "In this paper", "Deep learning"]
    for prompt in prompts:
        trainer.generate_sample(prompt)
    
    return history


def train_seq2seq(
    papers: List[Dict] = None,
    d_model: int = 256,
    num_heads: int = 4,
    num_layers: int = 4,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 3e-4
) -> Dict:
    """
    Train the Encoder-Decoder summarization model.
    
    Returns training history with ROUGE scores.
    """
    print("\n" + "="*60)
    print("📄 TRAINING ENCODER-DECODER SUMMARIZER")
    print("="*60)
    
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
    
    # Load data
    if papers is None:
        papers = load_papers()
    print(f"Loaded {len(papers)} papers")
    
    # Load or train tokenizer
    try:
        tokenizer = load_tokenizer()
    except FileNotFoundError:
        texts = [p['abstract'] for p in papers]
        tokenizer = train_bpe_tokenizer(texts, vocab_size=8000)
    
    vocab_size = tokenizer.get_vocab_size()
    
    # Split data
    train_papers, val_papers = split_data(papers.copy())
    
    # Create datasets
    train_dataset = SummarizationDataset(train_papers, tokenizer)
    val_dataset = SummarizationDataset(val_papers, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model with increased dropout
    model = EncoderDecoderTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        dropout=0.2  # Increased for better generalization
    )
    print(f"Model: {count_parameters(model):,} parameters")
    
    # Train
    trainer = Seq2SeqTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        lr=lr,
        epochs=epochs,
        log_dir=LOGS_DIR
    )
    
    history = trainer.train()
    
    # Generate samples
    print("\n📝 Sample Summaries:")
    trainer.generate_sample()
    
    return history


def train_all_models(papers: List[Dict] = None) -> Dict:
    """
    Train all three transformer variants and return combined results.
    """
    results = {}
    
    # 1. Classifier
    results['classifier'] = train_classifier(papers)
    
    # 2. Language Model
    results['lm'] = train_language_model(papers)
    
    # 3. Seq2Seq
    results['seq2seq'] = train_seq2seq(papers)
    
    print("\n" + "="*60)
    print("🎉 ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*60)
    print(f"Checkpoints saved to: {CHECKPOINTS_DIR}")
    print(f"Logs saved to: {LOGS_DIR}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 1 Training")
    parser.add_argument('--model', choices=['classifier', 'lm', 'seq2seq', 'all'], 
                        default='all', help="Model to train")
    parser.add_argument('--epochs', type=int, default=30, help="Number of epochs")
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size")
    parser.add_argument('--d-model', type=int, default=256, help="Model dimension")
    parser.add_argument('--num-heads', type=int, default=4, help="Number of attention heads")
    parser.add_argument('--num-layers', type=int, default=4, help="Number of layers")
    
    args = parser.parse_args()
    
    # Load data once
    papers = load_papers()
    
    if args.model == 'classifier':
        train_classifier(papers, args.d_model, args.num_heads, args.num_layers, args.epochs, args.batch_size)
    elif args.model == 'lm':
        train_language_model(papers, args.d_model, args.num_heads, args.num_layers, args.epochs, args.batch_size)
    elif args.model == 'seq2seq':
        train_seq2seq(papers, args.d_model, args.num_heads, args.num_layers, args.epochs, args.batch_size)
    else:
        train_all_models(papers)
