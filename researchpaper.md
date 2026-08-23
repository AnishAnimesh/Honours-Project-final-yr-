# How to Write Your Research Paper
## A Step-by-Step Guide for Complete Beginners

Good news: you're in a much better position than most people starting a research paper — you already have almost everything you need (problem statement, literature review, results, a case study). Writing the paper now is mostly about **organizing what you already have into the right structure**, not creating new content from scratch.

---

## 1. The Standard Structure (IMRAD)

Nearly all research papers follow the same skeleton, called **IMRAD**:

| Section | What it means | What goes here |
|---|---|---|
| **I** — Introduction | Why does this matter? | Problem, motivation, objectives |
| **M** — Methods | What did you do? | Data, model, methodology |
| **R** — Results | What did you find? | Tables, numbers, case study |
| **A** — And | (silent, just connects) | — |
| **D** — Discussion | What does it mean? | Interpretation, limitations, conclusion |

Plus a few extra pieces around this core: **Title, Abstract, Literature Review, References.**

Here's the full paper structure in order:

1. Title
2. Abstract
3. Introduction
4. Literature Review / Related Work
5. Methodology
6. Results
7. Discussion
8. Conclusion & Future Work
9. References

---

## 2. Section-by-Section: What to Write and Where You Already Have It

### **Title**
Use what we already crafted: *"Deep Learning-Based Forecasting of European Heat Waves Using ERA5 Reanalysis Data"*

### **Abstract** (150-250 words)
A mini-summary of the entire paper, written **last** (after everything else is done). One sentence for each of: problem, method, key result, conclusion. Example skeleton:

> "Heat waves in Europe have become more frequent and severe due to climate change, yet accurate early prediction remains challenging. This study applies deep learning — specifically LSTM and ConvLSTM architectures — to forecast heat wave occurrence using ERA5 reanalysis data (temperature, pressure, soil moisture) across a European region from 2020-2026. The ConvLSTM model achieved an AUC of 0.851, substantially outperforming a single-location LSTM (AUC 0.729) and simple baselines. A case study on the August 2026 heat wave revealed that while the model localizes moderate events well, it underpredicts unprecedented, large-scale events — highlighting a key limitation of historical data-driven approaches in an accelerating climate. These findings [contribute to / support] the growing body of work applying deep learning to climate early-warning systems."

### **Introduction** (~1-1.5 pages)
You already have this — it's your **problem statement**, expanded slightly:
- Open with why heat waves matter (climate change context, 2026 events)
- State the problem (hard to predict, especially at longer lead times)
- State your objective/research question clearly
- Briefly preview your approach and key finding (1-2 sentences)
- End with a one-sentence "roadmap": *"The remainder of this paper is organized as follows: Section 2 reviews related work, Section 3 describes the methodology..."*

### **Literature Review** (~1-2 pages)
You already have this too — it's your **literature review document**, expanded into paragraph form (not bullet points, for a formal paper). Organize by theme, not by paper:
- Paragraph 1: Statistical/feature-selection approaches (the Nature paper)
- Paragraph 2: ConvLSTM/spatiotemporal approaches (your closest architecture reference)
- Paragraph 3: Subseasonal prediction approaches
- Paragraph 4: The gap your paper addresses (real-event case study testing, which is less common)

### **Methodology** (~1.5-2 pages)
You already have all of this from your notebook work — just describe it in prose instead of code:
- **Data:** ERA5 source, variables used, region, time period, how it was cleaned
- **Heat wave definition:** the 90th percentile + 3-day rule, with justification
- **Models:** describe LSTM architecture (input window, hidden units, layers), then ConvLSTM architecture (same, plus how it handles spatial data)
- **Training details:** loss function (BCEWithLogitsLoss with class weighting), optimizer, train/val/test split strategy (and why you split by time, not randomly)
- **Evaluation metrics:** accuracy, F1, AUC — and *why* accuracy alone is insufficient (your climatology baseline finding is great justification here)

### **Results** (~1.5-2 pages)
You already have this — it's your comparison table and case study, described in prose + tables/figures:
- Present the baseline comparison table
- Present the ablation study table
- Present the case study (August 2026) with your saved figures
- **Important: in this section, only report findings — don't interpret them yet.** ("The ConvLSTM achieved an AUC of 0.851, compared to 0.729 for the LSTM" — not "this proves ConvLSTM is better because...")

### **Discussion** (~1-1.5 pages)
This is where you **interpret** the results — the "so what?" section:
- Why does ConvLSTM outperform LSTM? (spatial context, more training examples)
- Why does persistence beat both deep learning models? (heat waves are multi-day by nature)
- Why did the model fail on August 4th specifically? (unprecedented event, outside training distribution — tie this back to your literature review's identified gap)
- Limitations (small dataset, limited years, compute constraints, region size)

### **Conclusion & Future Work** (~0.5 page)
- Restate your research question and answer it directly in 1-2 sentences
- Summarize the key finding (ConvLSTM works reasonably well for typical events, struggles with extremes)
- List 2-3 concrete future directions (more years of data, transformer architectures, coastal region with SST)

### **References**
List every paper from your literature review in proper citation format (ask your guide which style — IEEE, APA, or your university's required format). Include the ERA5 paper (Hersbach et al., 2020) since you cited that dataset throughout.

---

## 3. Practical Writing Tips (Since You're New to This)

1. **Write sections in this order, not top-to-bottom:** Methods → Results → Discussion → Introduction → Literature Review → Conclusion → Abstract. It's much easier to write the Introduction once you already know exactly what you found.

2. **Every claim needs support.** Don't just say "the model performed well" — say "the model achieved an AUC of 0.851, indicating strong discriminative ability." Numbers, not adjectives.

3. **Use past tense for what you did** ("We trained a ConvLSTM model...") and **present tense for established facts** ("Heat waves are associated with increased mortality...").

4. **Figures and tables need captions and in-text references.** Every table/figure should be numbered (Table 1, Figure 1) and mentioned in the text ("As shown in Table 1, the ConvLSTM outperformed...").

5. **Avoid "I" — use "we" or passive voice** in most academic writing (check your guide's preference, some fields now accept "I" more).

6. **Don't hide weak results — explain them.** Your August 4th failure case is actually one of the *strongest* parts of your paper if written well, because it shows real scientific understanding, not just "my model works."

7. **Length:** ask your guide for an expected page count — this varies hugely by program (could be 6 pages, could be 20+). Everything above scales up or down based on that.

---

## 4. Suggested Order of Work

1. Write **Methodology** first (easiest — just describe what you already built)
2. Write **Results** next (also easy — just present your tables/figures)
3. Write **Discussion** (interpret what you found — use the explanations we've already worked through together)
4. Write **Introduction** (now that you know your full story)
5. Expand your **Literature Review** into paragraph form
6. Write **Conclusion**
7. Write **Abstract** last
8. Format **References** properly
9. Proofread the whole thing top to bottom

---

## What I Can Help With Next

- Draft any individual section in full prose, using everything we've built together
- Convert your literature review bullet points into proper academic paragraphs
- Help format references in a specific citation style
- Review a draft you write yourself and suggest improvements

Want to start by drafting the Methodology section together, since that's usually the easiest one to begin with?