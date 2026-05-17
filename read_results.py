import json
with open('results/benchmark_20260515_144550.json', 'r') as f:
    data = json.load(f)

print('=== BENCHMARK SUMMARY ===')
for pk in ['llm_only', 'basic_rag', 'graphrag']:
    d = data.get(pk, {})
    j = data.get('judge', {}).get(pk, {})
    b = data.get('bertscore', {}).get(pk, {})
    if d:
        print(f'{pk.upper()}:')
        print(f'  Avg Tokens: {d.get("avg_tokens")}')
        print(f'  Avg Latency: {d.get("avg_latency")}s')
        print(f'  Pass Rate: {j.get("pass_rate")}')
        print(f'  BERTScore F1: {b.get("avg_f1_rescaled")}')
        
print('\n=== COMPARISON ===')
print(json.dumps(data.get('comparison', {}), indent=2))
