---
name: wave_hello
keywords: [wave, hello, hi, greet, greeting]
params: {theta: 0.8}
---
# Template
{"title": "wave hello", "duration_s": 3.0,
 "actions": [
   {"robot": "a", "t": 0.0, "tool": "joint", "params": {"theta": {{theta}}}},
   {"robot": "a", "t": 1.0, "tool": "reach", "params": {"x": 0.3, "y": 1.6}},
   {"robot": "a", "t": 2.0, "tool": "joint", "params": {"theta": -0.4}}
 ]}
