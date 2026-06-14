# ============================================================
# GenAI Hands-On: Sentiment, Spam Detection & Semantic Search
# ============================================================
# Run: pip install transformers torch sentence-transformers
# Then: python3 genai_hands_on.py
# ============================================================

from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import numpy as np

# ────────────────────────────────────────────────────────────
# 1. SENTIMENT ANALYSIS (Encoder - BERT)
# ────────────────────────────────────────────────────────────
# Under the hood:
#   Input → BERT encoder → [CLS] embedding (768 dims)
#   → W_head (768 → 768) → W_classifier (768 → 2) → softmax → positive/negative
#
# HuggingFace wraps all of this in one line.
# ────────────────────────────────────────────────────────────

print("=" * 60)
print("1. SENTIMENT ANALYSIS")
print("=" * 60)

sentiment = pipeline("sentiment-analysis")

texts = [
    "I absolutely loved this movie, it was brilliant!",
    "The food was terrible and the service was worse.",
    "It was okay, nothing special but not bad either.",
    "Bangalore weather is amazing during October!",
    "The new iPhone is a complete waste of money.",
]

print("\nResults:")
for text in texts:
    result = sentiment(text)[0]
    print(f"  '{text[:50]}...'")
    print(f"    → {result['label']} (confidence: {result['score']:.2%})")
    print()

# ─── Inspect the actual matrix dimensions ─────────────────────
print("-" * 60)
print("SENTIMENT MODEL — MATRIX DIMENSIONS")
print("-" * 60)
m = sentiment.model

word_emb = m.distilbert.embeddings.word_embeddings.weight
pos_emb = m.distilbert.embeddings.position_embeddings.weight
print(f"\n  EMBEDDING TABLE:")
print(f"    word_embeddings:     {tuple(word_emb.shape)}  <- {word_emb.shape[0]} vocab tokens x {word_emb.shape[1]} dims")
print(f"    position_embeddings: {tuple(pos_emb.shape)}  <- max {pos_emb.shape[0]} positions x {pos_emb.shape[1]} dims")

layer0 = m.distilbert.transformer.layer[0]
wq = layer0.attention.q_lin.weight
wk = layer0.attention.k_lin.weight
wv = layer0.attention.v_lin.weight
wo = layer0.attention.out_lin.weight
print(f"\n  ATTENTION (Layer 0) — 12 heads x {wq.shape[0]//12} dims each:")
print(f"    W_q:  {tuple(wq.shape)}  <- input {wq.shape[1]} -> output {wq.shape[0]} (12 heads x {wq.shape[0]//12})")
print(f"    W_k:  {tuple(wk.shape)}  <- input {wk.shape[1]} -> output {wk.shape[0]} (12 heads x {wk.shape[0]//12})")
print(f"    W_v:  {tuple(wv.shape)}  <- input {wv.shape[1]} -> output {wv.shape[0]} (12 heads x {wv.shape[0]//12})")
print(f"    W_o:  {tuple(wo.shape)}  <- concatenated {wo.shape[1]} -> mixed {wo.shape[0]}")

ff1 = layer0.ffn.lin1.weight
ff2 = layer0.ffn.lin2.weight
print(f"\n  FEED-FORWARD (Layer 0):")
print(f"    lin1:  {tuple(ff1.shape)}  <- expand {ff1.shape[1]} -> {ff1.shape[0]}")
print(f"    lin2:  {tuple(ff2.shape)}  <- compress {ff2.shape[1]} -> {ff2.shape[0]}")

pre_w = m.pre_classifier.weight
cls_w = m.classifier.weight
print(f"\n  CLASSIFICATION HEAD:")
print(f"    pre_classifier:  {tuple(pre_w.shape)}  <- {pre_w.shape[1]} -> {pre_w.shape[0]}")
print(f"    classifier:      {tuple(cls_w.shape)}    <- {cls_w.shape[1]} -> {cls_w.shape[0]} classes (pos/neg)")

num_layers = len(m.distilbert.transformer.layer)
total_params = sum(p.numel() for p in m.parameters())
print(f"\n  SUMMARY:")
print(f"    Encoder layers: {num_layers}")
print(f"    Total parameters: {total_params:,}")
print()


# ────────────────────────────────────────────────────────────
# 2. SPAM DETECTION (Encoder - BERT)
# ────────────────────────────────────────────────────────────
# Same architecture as sentiment, just different training data:
#   Input → BERT encoder → [CLS] embedding
#   → W_head → softmax → spam/ham
#
# Remember: the head is what changes, not the architecture.
# ────────────────────────────────────────────────────────────

print("=" * 60)
print("2. SPAM DETECTION")
print("=" * 60)

spam_detector = pipeline(
    "text-classification",
    model="mrm8488/bert-tiny-finetuned-sms-spam-detection"
)

messages = [
    "Congratulations! You've won a FREE iPhone! Click here NOW!",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT: Your bank account has been compromised. Send details.",
    "Can you pick up groceries on the way home?",
    "You have been selected for a $1000 gift card. Act fast!",
    "Meeting pushed to 3pm, see you in the conf room.",
]

print("\nResults:")
for msg in messages:
    result = spam_detector(msg)[0]
    label = "SPAM" if result["label"] == "LABEL_1" else "HAM"
    print(f"  '{msg[:55]}...'")
    print(f"    -> {label} (confidence: {result['score']:.2%})")
    print()

# ─── Inspect spam model dimensions ──────────────────────────
print("-" * 60)
print("SPAM MODEL — MATRIX DIMENSIONS")
print("-" * 60)
sm = spam_detector.model

word_emb = sm.bert.embeddings.word_embeddings.weight
pos_emb = sm.bert.embeddings.position_embeddings.weight
print(f"\n  EMBEDDING TABLE:")
print(f"    word_embeddings:     {tuple(word_emb.shape)}  <- {word_emb.shape[0]} vocab x {word_emb.shape[1]} dims")
print(f"    position_embeddings: {tuple(pos_emb.shape)}  <- max {pos_emb.shape[0]} positions x {pos_emb.shape[1]} dims")

layer0 = sm.bert.encoder.layer[0]
wq = layer0.attention.self.query.weight
wk = layer0.attention.self.key.weight
wv = layer0.attention.self.value.weight
wo = layer0.attention.output.dense.weight
num_heads = layer0.attention.self.num_attention_heads
head_dim = wq.shape[0] // num_heads
print(f"\n  ATTENTION (Layer 0) — {num_heads} heads x {head_dim} dims each:")
print(f"    W_q:  {tuple(wq.shape)}  <- input {wq.shape[1]} -> output {wq.shape[0]}")
print(f"    W_k:  {tuple(wk.shape)}  <- input {wk.shape[1]} -> output {wk.shape[0]}")
print(f"    W_v:  {tuple(wv.shape)}  <- input {wv.shape[1]} -> output {wv.shape[0]}")
print(f"    W_o:  {tuple(wo.shape)}  <- concatenated {wo.shape[1]} -> mixed {wo.shape[0]}")

ff1 = layer0.intermediate.dense.weight
ff2 = layer0.output.dense.weight
print(f"\n  FEED-FORWARD (Layer 0):")
print(f"    lin1:  {tuple(ff1.shape)}  <- expand {ff1.shape[1]} -> {ff1.shape[0]}")
print(f"    lin2:  {tuple(ff2.shape)}  <- compress {ff2.shape[1]} -> {ff2.shape[0]}")

cls_w = sm.classifier.weight
print(f"\n  CLASSIFICATION HEAD:")
print(f"    classifier:  {tuple(cls_w.shape)}  <- {cls_w.shape[1]} -> {cls_w.shape[0]} classes (spam/ham)")

num_layers = len(sm.bert.encoder.layer)
total_params = sum(p.numel() for p in sm.parameters())
print(f"\n  SUMMARY:")
print(f"    Encoder layers: {num_layers}")
print(f"    Total parameters: {total_params:,}")
print(f"\n  Notice how much smaller this is than the sentiment model!")
print(f"  'bert-tiny' = fewer layers, smaller dims. Faster but less accurate.")
print()


# ────────────────────────────────────────────────────────────
# FINAL: MODEL PARAMETER COMPARISON
# ────────────────────────────────────────────────────────────

def param_breakdown(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    size_mb = total * 4 / (1024 ** 2)  # float32 = 4 bytes
    return total, trainable, frozen, size_mb

print("=" * 60)
print("FINAL: MODEL PARAMETER COMPARISON")
print("=" * 60)

models = {
    "Sentiment (DistilBERT)": sentiment.model,
    "Spam Detector (BERT-tiny)": spam_detector.model,
}

print(f"\n  {'Model':<30} {'Total':>12} {'Trainable':>12} {'Frozen':>10} {'Size (MB)':>10}")
print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")

for name, model in models.items():
    total, trainable, frozen, size_mb = param_breakdown(model)
    print(f"  {name:<30} {total:>12,} {trainable:>12,} {frozen:>10,} {size_mb:>9.1f}")

print()
sent_total, *_ = param_breakdown(sentiment.model)
spam_total, *_ = param_breakdown(spam_detector.model)
print(f"  Sentiment model is {sent_total / spam_total:.1f}x larger than Spam model.")
print(f"  More params = richer representations, but slower and heavier.")
print()

