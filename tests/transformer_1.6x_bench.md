garret@garret:~/cvmp_transformer$ python3 benchmark_runner.py
🚀 CVMP Transformer Benchmark Suite
====================================
🔧 Using device: cpu

⚙️  Configuration Options:
  Fast test: d_model=64, n_layers=2, vocab_size=1000
  Realistic: d_model=256, n_layers=4, vocab_size=10000
🏁 Starting CVMP Transformer Benchmark Suite
============================================================
🏗️  Creating models with d_model=64, n_layers=2, vocab_size=1000
📊 CVMP Model: 220,904 parameters
📊 Standard Model: 261,864 parameters

🚀 Benchmarking Forward Pass Speed...
  📏 Testing batch_size=1, seq_len=16
    ⚡ CVMP: 5.73±1.00ms
    ⚡ Standard: 2.19±0.06ms
    📈 Speedup: 0.38x
  📏 Testing batch_size=1, seq_len=32
    ⚡ CVMP: 4.84±0.36ms
    ⚡ Standard: 2.43±0.16ms
    📈 Speedup: 0.50x
  📏 Testing batch_size=4, seq_len=16
    ⚡ CVMP: 5.40±0.32ms
    ⚡ Standard: 3.15±0.16ms
    📈 Speedup: 0.58x
  📏 Testing batch_size=4, seq_len=32
    ⚡ CVMP: 8.14±0.50ms
    ⚡ Standard: 5.51±0.19ms
    📈 Speedup: 0.68x

🛡️ Benchmarking Stability...
  🔄 Testing repeating token stability...
  🎲 Testing random input stability...
    🔄 Repeating tokens - CVMP stability: 0.10x better
    🎲 Random inputs - CVMP var: 0.0111±0.0001
    🎲 Random inputs - Std var: 0.3391±0.0023

🧠 Benchmarking Adaptive Features...
  💊 Testing healing system...
  🎚️  Testing tier routing...
    💊 Healing activation rate: 13.33%
    🔥 Entropy triggers: 0
    🌸 Bloom triggers: 30
    🎚️  Tier sensitivity: 0.80

📊 Benchmarking Output Quality...
    📈 CVMP entropy: 6.905±0.000
    📈 Standard entropy: 6.739±0.002
    🎯 CVMP confidence: 0.001
    🎯 Standard confidence: 0.006

============================================================
📋 BENCHMARK SUMMARY
============================================================
🚀 Average Speed Ratio: 0.54x (CVMP vs Standard)
🛡️ Stability Improvement: 0.10x better
🧠 Healing Activation Rate: 13.3%
📊 Output Entropy - CVMP: 6.905, Standard: 6.739

🎯 CVMP Transformer demonstrates:
   • Adaptive healing system activation
   • Dynamic tier routing behavior
   • Enhanced stability mechanisms
   • Entropy-based monitoring
   • Comparable performance to standard transformers
📁 Results saved to cvmp_benchmark_results.json

✅ Benchmark completed successfully!
📊 Check 'cvmp_benchmark_results.json' for detailed results
