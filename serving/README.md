# Requirements

Requirements depend mostly on the type of model you want to use and its
optimizations. A conservative rule of thumb is to have enough GPU memory
to fit the model given by the expression ([some](https://blog.eleuther.ai/transformer-math/#total-inference-memory)
recommend adding only 20% extra and not 35% as below):

---
**GPU [GB]** = Model Parameters [Billion] × Precision [bits]/8 × 1.35
---
