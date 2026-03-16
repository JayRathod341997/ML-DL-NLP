# GRU Interview Questions

## Q1: What is the difference between GRU and LSTM?

**Answer:** GRU has 2 gates (update, reset) vs LSTM's 3 gates (input, forget, output). GRU doesn't have a separate cell state. GRU has fewer parameters and is faster but may not capture all long-term dependencies as well.

---

## Q2: When should you choose GRU over LSTM?

**Answer:** Choose GRU when: training speed matters, you have limited data, or sequence lengths are moderate. Use LSTM when you need maximum flexibility or have very long sequences.

---

## Q3: Explain the update gate in GRU.

**Answer:** The update gate (z_t) controls how much of the previous hidden state to keep. When z_t is close to 1, we keep most of the past; when close to 0, we reset to new memory.

---

## Q4: What is the reset gate used for?

**Answer:** The reset gate (r_t) determines how much of the past hidden state to ignore when computing new memory. A low reset value allows the network to "forget" irrelevant past information.

---

## Q5: Why is GRU faster than LSTM?

**Answer:** GRU has fewer matrix multiplications because it has 2 gates instead of 3, and no cell state. This reduces both computation and memory requirements.
