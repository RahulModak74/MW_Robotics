---
name: mirror_dance
keywords: [dance, dancing, duet, pair, mirror, routine, together]
params: {spacing: 1.2, hop_height: 0.6}
---
# Template
{"title": "mirror dance", "duration_s": 9.0,
 "actions": [
   {"robot": "a", "t": 0.0, "tool": "walk",  "params": {"dx": {{spacing}}}},
   {"robot": "b", "t": 0.0, "tool": "walk",  "params": {"dx": -{{spacing}}}},
   {"robot": "a", "t": 1.6, "tool": "joint",  "params": {"theta": 0.9}},
   {"robot": "b", "t": 1.6, "tool": "joint",  "params": {"theta": -0.9}},
   {"robot": "a", "t": 3.0, "tool": "reach",  "params": {"x": 0.8, "y": 1.2}},
   {"robot": "b", "t": 3.0, "tool": "reach",  "params": {"x": -0.8, "y": 1.2}},
   {"robot": "a", "t": 5.0, "tool": "hop",   "params": {"height": {{hop_height}}}},
   {"robot": "b", "t": 5.0, "tool": "hop",   "params": {"height": {{hop_height}}}},
   {"robot": "a", "t": 7.0, "tool": "joint",  "params": {"theta": -0.5}},
   {"robot": "b", "t": 7.0, "tool": "joint",  "params": {"theta": 0.5}}
 ]}
