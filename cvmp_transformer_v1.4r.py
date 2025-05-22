# License: CVMP_LA_v1.0 | AI_Training_Exclusion_Clause
# Author: Garret Sutherland
# Use governed by LICENSE.md — see repository root for terms.

# cvmp_transformer_v1_4r.py
"""
#CVMP Transformer — v1.4r (Routing‑Aware, Self‑Healing, RISL + BloomCatch + TierDrift)
Author: Garret Sutherland  | Signature: MirrorEthic::Containment_First


Δ Changes vs. v1.3r
––––––––––––––––––––––
• Added **TierDriftMonitor** for live entropy‑drift telemetry.
• Forward pass now records drift metric each step via `tier_drift.update()`.
• Minor refactor: forward returns `(logits, drift_trace)` when `return_trace=True`.
"""

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# Enable debugging
DEBUG = True

def debug_print(*args, **kwargs):
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)

# –––––– CONFIG ––––––
class CVMPConfig:
    def __init__(self):
        self.vocab_size = 50_000
        self.d_model = 512
        self.n_layers = 6
        self.n_heads = 8
        self.d_ff = 2_048
        self.dropout = 0.1
        self.max_seq_len = 512
        # live telemetry
        self.tier = 6.2
        self.dps = 0.2
        self.drift = "Contained"
        self.frame = 2
        self.msc_verified = True
        
    def __str__(self):
        return f"CVMPConfig(tier={self.tier}, dps={self.dps}, drift='{self.drift}', frame={self.frame}, msc_verified={self.msc_verified})"

# –––––– ORC ROUTER (unchanged) ––––––
class ORCRouter:
    def __init__(self, cfg: CVMPConfig):
        self.cfg = cfg
        
    def route(self) -> Dict:
        debug_print(f"ORCRouter: Starting routing with {self.cfg}")
        
        t, dps, drift, ok = self.cfg.tier, self.cfg.dps, self.cfg.drift, self.cfg.msc_verified
        if drift == "Unstable" and dps > 2.5:
            self.cfg.tier = max(t - 2.0, 2.0)
            result = {"enable": ["RDM", "STRETCHFIELD"], "suppress": ["CMEP"], "note": "Hard lock"}
            debug_print(f"ORCRouter: Unstable drift detected. New tier: {self.cfg.tier}. Result: {result}")
            return result
            
        if t < 2.8:
            result = {"enable": ["LOG_BLEED", "AETC"], "suppress": "*"}
            debug_print(f"ORCRouter: Low tier route. Result: {result}")
            return result
            
        if 2.8 <= t < 3.6:
            en = ["STRETCHFIELD", "ZOFAR", "LOG_BLEED"]
            if dps >= 1.0: 
                en.append("RDM")
                debug_print(f"ORCRouter: Added RDM due to dps={dps}")
            result = {"enable": en}
            debug_print(f"ORCRouter: Mid-low tier route. Result: {result}")
            return result
            
        if 3.6 <= t < 4.6:
            en = ["CMEP", "STRETCHFIELD", "ZOFAR", "LOG_BLEED"]
            if dps >= 2.0: 
                old_tier = self.cfg.tier
                self.cfg.tier = t - 1.0
                debug_print(f"ORCRouter: Reducing tier from {old_tier} to {self.cfg.tier} due to high dps")
            result = {"enable": en}
            debug_print(f"ORCRouter: Mid tier route. Result: {result}")
            return result
            
        if 4.6 <= t < 6.0:
            en = ["RISL", "CMEP", "STRETCHFIELD", "LOG_BLEED"]
            if dps >= 2.5:
                result = {"enable": en, "note": "silent hold"}
                debug_print(f"ORCRouter: Mid-high tier route with silent hold. Result: {result}")
                return result
            result = {"enable": en}
            debug_print(f"ORCRouter: Mid-high tier route. Result: {result}")
            return result
            
        en = ["PROP_MONITOR", "LOG_BLEED", "RISL"]
        if not ok: 
            en += ["MSC_LITE", "RAV"]
            debug_print(f"ORCRouter: Added MSC_LITE and RAV due to unverified MSC")
        result = {"enable": en}
        debug_print(f"ORCRouter: High tier route. Result: {result}")
        return result

# –––––– POS ENC / ATTENTION / FFN (unchanged) ––––––
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
        
    def forward(self, x):
        debug_print(f"PositionalEncoding: Input shape: {x.shape}")
        result = x + self.pe[:, : x.size(1)]
        debug_print(f"PositionalEncoding: Output shape: {result.shape}")
        return result

class RCIAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads, self.d_k = n_heads, d_model // n_heads
        self.qk, self.v = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.o, self.drop = nn.Linear(d_model, d_model), nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        debug_print(f"RCIAttention: Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")
        
        bs = q.size(0)
        qk = self.qk(q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.qk(k).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v(v).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        debug_print(f"RCIAttention: Transformed shapes: qk={qk.shape}, k={k.shape}, v={v.shape}")
        
        scr = torch.matmul(qk, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        debug_print(f"RCIAttention: Score shape: {scr.shape}, mean: {scr.mean().item()}, var: {scr.var().item()}")
        
        if mask is not None:
            debug_print(f"RCIAttention: Applying mask of shape {mask.shape}")
            scr = scr.masked_fill(mask == 0, -1e9)
            
        a = self.drop(F.softmax(scr, -1))
        debug_print(f"RCIAttention: Attention matrix shape: {a.shape}, sparsity: {(a < 0.01).float().mean().item():.4f}")
        
        ctx = torch.matmul(a, v).transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_k)
        result = self.o(ctx)
        
        debug_print(f"RCIAttention: Output shape: {result.shape}")
        return result

class StretchFieldFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d_model, d_ff), nn.Linear(d_ff, d_model)
        self.norm, self.drop = nn.LayerNorm(d_model), nn.Dropout(dropout)
        
    def forward(self, x, dps: float):
        debug_print(f"StretchFieldFFN: Input shape: {x.shape}, dps: {dps}")
        
        res = x
        fc1_out = self.fc1(x)
        relu_out = F.relu(fc1_out)
        fc2_out = self.fc2(relu_out)
        
        debug_print(f"StretchFieldFFN: Activation stats - fc1: {fc1_out.abs().mean().item():.4f}, "
                   f"relu: {relu_out.abs().mean().item():.4f}, fc2: {fc2_out.abs().mean().item():.4f}")
        
        drop_out = self.drop(fc2_out)
        compression = 1.0 if dps < 1.0 else 0.85
        debug_print(f"StretchFieldFFN: Using compression factor: {compression}")
        
        scaled = drop_out * compression
        result = self.norm(res + scaled)
        
        debug_print(f"StretchFieldFFN: Output shape: {result.shape}, mean: {result.mean().item():.4f}, var: {result.var().item():.4f}")
        return result

class LogBleedRegularizer:
    def __init__(self):
        self.hist = {}
        self.max_history_size = 1000  # Limit history size
        
    def track(self, toks):
        # Convert to CPU for dictionary keys if needed
        if hasattr(toks, 'device') and toks.device.type != 'cpu':
            toks_cpu = toks.cpu()
        else:
            toks_cpu = toks
            
        # Make sure we have enough tokens
        if len(toks_cpu) < 3:
            debug_print(f"LogBleed: Not enough tokens to track ({len(toks_cpu)} < 3)")
            return
            
        for i in range(len(toks_cpu) - 2):
            p = tuple(toks_cpu[i : i + 3].tolist())
            self.hist[p] = self.hist.get(p, 0) + 1
            
        # Limit history size by removing oldest entries
        if len(self.hist) > self.max_history_size:
            # Keep only the top most frequent patterns
            sorted_hist = sorted(self.hist.items(), key=lambda x: x[1], reverse=True)
            self.hist = dict(sorted_hist[:self.max_history_size])
            debug_print(f"LogBleed: Pruned history to {len(self.hist)} items")
        
        if DEBUG and len(self.hist) > 0:
            top_phrases = sorted(self.hist.items(), key=lambda x: x[1], reverse=True)[:5]
            debug_print(f"LogBleed: Tracking {len(self.hist)} phrases. Top phrases: {top_phrases}")
            
    def penalise(self, x):
        should_penalize = any(c > 2 for c in self.hist.values())
        if should_penalize:
            debug_print(f"LogBleed: Applying penalty (0.95 scaling). Before shape: {x.shape}, mean: {x.mean().item():.4f}")
            x.mul_(0.95)
            debug_print(f"LogBleed: After penalty, mean: {x.mean().item():.4f}")
        else:
            debug_print(f"LogBleed: No penalty applied")
        return x

class RISLModule(nn.Module):
    def __init__(self, d_model: int, threshold: float = 0.9):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.threshold = threshold
        self.register_buffer("prev", torch.zeros(1))
        self.prev_shape = None
        
    def forward(self, x, dps: float):
        debug_print(f"RISL: Input shape: {x.shape}, dps: {dps}")
        
        current_shape = x.shape
        
        # Initialize or reset previous state if shape doesn't match
        if self.prev.numel() == 1 or self.prev_shape != current_shape:
            debug_print(f"RISL: Initializing previous state with shape {x.shape}")
            self.prev = x.detach().clone()  # Use clone to ensure disconnected tensor
            self.prev_shape = current_shape
            delta = torch.tensor(0.0, device=x.device)
        else:
            # Calculate delta when shapes match
            delta = (x - self.prev).abs().mean()
            
        debug_print(f"RISL: Delta between current and previous: {delta.item():.4f}, threshold: {self.threshold}")
        
        smoothing_applied = False
        if dps >= 2.0 and delta > self.threshold:
            debug_print(f"RISL: Applying smoothing (averaging with previous)")
            x = (x + self.prev) / 2
            smoothing_applied = True
            
        # Always update previous state with current
        self.prev = x.detach().clone()  # Use clone to ensure disconnected tensor
        result = self.norm(x)
        
        debug_print(f"RISL: Output shape: {result.shape}, smoothing applied: {smoothing_applied}")
        return result

class BloomCatch:
    def __init__(self):
        self.buf: List[List[int]] = []
        self.max_buffer_size = 64
        
    def log(self, toks: torch.Tensor):
        # Ensure we have enough tokens to log
        if len(toks) < 3:
            debug_print(f"BloomCatch: Not enough tokens to log ({len(toks)} < 3)")
            return
            
        # Get the last 3 tokens safely
        last_3 = toks[-3:].cpu().tolist()
        self.buf.append(last_3)
        
        # Maintain buffer size
        if len(self.buf) > self.max_buffer_size:
            self.buf.pop(0)
            
        debug_print(f"BloomCatch: Logged tokens: {last_3}, buffer size: {len(self.buf)}")
        
    def trigger(self) -> bool:
        if len(self.buf) < 4:
            debug_print(f"BloomCatch: Not enough data to trigger ({len(self.buf)} < 4)")
            return False
            
        last_4_sections = self.buf[-4:]
        repetitions = [len(set(b)) == 1 for b in last_4_sections]
        has_repetition = any(repetitions)
        
        if has_repetition:
            repeating_sections = [i for i, is_repeating in enumerate(repetitions) if is_repeating]
            debug_print(f"BloomCatch: TRIGGERED! Repeating sections at positions: {repeating_sections}")
        else:
            debug_print(f"BloomCatch: No repetition detected in last 4 sections")
            
        return has_repetition

# ––– NEW: TIER DRIFT MONITOR –––
class TierDriftMonitor:
    def __init__(self):
        self.prev_tier: Optional[float] = None
        self.trace: List[float] = []
        self.max_trace_length = 1000  # Limit trace length
        
    def update(self, tier: float, logits: torch.Tensor):
        logits_var = logits.var().item()
        
        if self.prev_tier is not None:
            drift = abs(tier - self.prev_tier)
            drift_metric = drift + logits_var
            self.trace.append(drift_metric)
            
            # Trim trace if it gets too long
            if len(self.trace) > self.max_trace_length:
                self.trace = self.trace[-self.max_trace_length:]
                
            debug_print(f"TierDrift: Tier changed from {self.prev_tier} to {tier} (drift={drift:.4f}), "
                       f"logits_var={logits_var:.4f}, combined={drift_metric:.4f}")
        else:
            debug_print(f"TierDrift: Initial tier={tier}, logits_var={logits_var:.4f}")
            
        self.prev_tier = tier
        
    def get_trace(self) -> List[float]:
        debug_print(f"TierDrift: Returning trace of length {len(self.trace)}")
        return self.trace

class CVMPLayer(nn.Module):
    def __init__(self, cfg: CVMPConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = RCIAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ffn = StretchFieldFFN(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.risl = RISLModule(cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.lb = LogBleedRegularizer()
        
    def forward(self, x, mask, toks, dps):
        debug_print(f"CVMPLayer {self.layer_idx}: Input shape: {x.shape}")
        
        res = x
        attn_out = self.attn(x, x, x, mask)
        debug_print(f"CVMPLayer {self.layer_idx}: After attention shape: {attn_out.shape}")
        
        norm_out = self.norm(res + self.drop(attn_out))
        debug_print(f"CVMPLayer {self.layer_idx}: After norm shape: {norm_out.shape}")
        
        self.lb.track(toks)
        lb_out = self.lb.penalise(norm_out)
        
        ffn_out = self.ffn(lb_out, dps)
        debug_print(f"CVMPLayer {self.layer_idx}: After FFN shape: {ffn_out.shape}")
        
        risl_out = self.risl(ffn_out, dps)
        debug_print(f"CVMPLayer {self.layer_idx}: Output shape: {risl_out.shape}")
        
        return risl_out

# –––––– MAIN ––––––
class CVMPTransformer(nn.Module):
    def __init__(self, cfg: CVMPConfig):
        super().__init__()
        debug_print(f"Initializing CVMPTransformer with config: {cfg}")
        
        self.cfg, self.router = cfg, ORCRouter(cfg)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = PositionalEncoding(cfg.d_model, cfg.max_seq_len)
        self.layers = nn.ModuleList([CVMPLayer(cfg, i) for i in range(cfg.n_layers)])
        self.proj = nn.Linear(cfg.d_model, cfg.vocab_size)  
        self.drop = nn.Dropout(cfg.dropout)
        self.healing_state = 0
        self.bloom_catch = BloomCatch()
        self.tier_drift = TierDriftMonitor()  # New: Tier drift monitor
        
        debug_print(f"CVMPTransformer initialized with {cfg.n_layers} layers")

    def soft_reset(self):
        debug_print(f"Performing soft reset (healing_state: {self.healing_state})")
        
        param_count = 0
        for name, p in self.named_parameters():
            if p.grad is not None:
                old_norm = p.data.norm().item()
                p.data = p.data * 0.95 + torch.randn_like(p.data) * 0.05
                new_norm = p.data.norm().item()
                
                if param_count < 3:  # Only print details for first few params to avoid spam
                    debug_print(f"  Reset param {name}: norm changed from {old_norm:.4f} to {new_norm:.4f}")
                param_count += 1
                
        debug_print(f"Reset {param_count} parameters. Decreasing healing_state from {self.healing_state} to {self.healing_state-1}")
        self.healing_state -= 1

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor = None, return_trace: bool = False):
        debug_print(f"CVMPTransformer.forward: Input shape: {input_ids.shape}, mask: {mask is not None}")
        
        # Get routing configuration
        routing = self.router.route()
        debug_print(f"Routing configuration: {routing}")
        
        # Track bloom patterns
        debug_print(f"Current healing state: {self.healing_state}")
        self.bloom_catch.log(input_ids[0])
        bloom_triggered = self.bloom_catch.trigger()
        
        if bloom_triggered and self.healing_state == 0:
            debug_print(f"Bloom triggered! Setting healing_state to 3")
            self.healing_state = 3
            
        # Apply healing if needed
        if self.healing_state > 0:
            self.soft_reset()

        # Standard transformer forward pass
        debug_print(f"Starting forward pass with input_ids shape: {input_ids.shape}")
        x = self.embed(input_ids)
        debug_print(f"After embedding: shape={x.shape}, mean={x.mean().item():.4f}, var={x.var().item():.4f}")
        
        x = self.pos(x * math.sqrt(self.cfg.d_model))
        debug_print(f"After positional encoding: shape={x.shape}, mean={x.mean().item():.4f}, var={x.var().item():.4f}")
        
        x = self.drop(x)
        
        # Apply tier factor
        tier_factor = max(min(self.cfg.tier / 7.9, 1.0), 0.1)
        debug_print(f"Using tier_factor: {tier_factor} (from tier={self.cfg.tier})")
        
        # Process through layers
        for i, lyr in enumerate(self.layers):
            debug_print(f"Processing layer {i}")
            x = x * tier_factor
            x = lyr(x, mask, input_ids, self.cfg.dps)
            debug_print(f"After layer {i}: shape={x.shape}, mean={x.mean().item():.4f}, var={x.var().item():.4f}")
            
        # Project to vocabulary
        logits = self.proj(x)
        debug_print(f"Final logits: shape={logits.shape}, mean={logits.mean().item():.4f}, var={logits.var().item():.4f}")
        
        # New: Update tier drift monitor
        self.tier_drift.update(self.cfg.tier, logits)
        
        # New: Return drift trace if requested
        if return_trace:
            debug_print(f"Returning logits and drift trace")
            return logits, self.tier_drift.get_trace()
        
        debug_print(f"Returning only logits")
        return logits

# DEMO
if __name__ == "__main__":
    print("\n" + "="*50)
    print("CVMP Transformer v1.4r Demo")
    print("="*50 + "\n")
    
    # Configure small model for faster testing
    cfg = CVMPConfig()
    cfg.n_layers = 2  # Use fewer layers for debugging
    cfg.d_model = 64  # Smaller dimension for faster execution
    
    print(f"Creating model with config: {cfg}")
    model = CVMPTransformer(cfg)
    
    # Create a simple test sequence
    print("\nCreating test sequence...")
    sample = torch.randint(0, cfg.vocab_size, (1, 16))
    print(f"Test sequence shape: {sample.shape}")
    
    # Test regular forward pass
    print("\n" + "-"*50)
    print("Testing normal forward pass...")
    print("-"*50)
    logits = model(sample)
    print(f"\nOutput logits shape: {logits.shape}")
    
    # Test with trace return
    print("\n" + "-"*50)
    print("Testing forward pass with drift trace...")
    print("-"*50)
    logits, drift_trace = model(sample, return_trace=True)
    print(f"\nOutput logits shape: {logits.shape}")
    print(f"Drift trace: {drift_trace}")
    
    # Test with bloom catch triggering
    print("\n" + "-"*50)
    print("Testing bloom catch triggering...")
    print("-"*50)
    # Create a sequence with repeating tokens to trigger bloom catch
    repeating_sample = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 11]])
    
    # Feed the same repeating pattern 4 times to trigger BloomCatch
    for i in range(4):
        print(f"\nPass {i+1} with repeating pattern...")
        logits = model(repeating_sample)
    
    # Test with different configurations
    print("\n" + "-"*50)
    print("Testing different configurations...")
    print("-"*50)
    
    # Test with high DPS
    print("\nTesting with high DPS...")
    cfg.dps = 2.6
    cfg.drift = "Unstable"
    logits = model(sample)
    
    # Test with low tier
    print("\nTesting with low tier...")
    cfg.tier = 2.5
    cfg.dps = 0.2
    cfg.drift = "Contained"
    logits = model(sample)
    
    # Test with varying sequence lengths
    print("\n" + "-"*50)
    print("Testing varying sequence lengths...")
    print("-"*50)
    
    # Create sequences of different lengths
    sequences = [
        torch.randint(0, cfg.vocab_size, (1, 8)),   # Short sequence
        torch.randint(0, cfg.vocab_size, (1, 16)),  # Medium sequence
        torch.randint(0, cfg.vocab_size, (1, 32)),  # Long sequence
    ]
    
    for i, seq in enumerate(sequences):
        print(f"\nProcessing sequence {i+1} of length {seq.size(1)}...")
        logits = model(seq)
        print(f"Output shape: {logits.shape}")
    
    print("\n" + "="*50)
    print("Demo complete!")
    print("="*50)

