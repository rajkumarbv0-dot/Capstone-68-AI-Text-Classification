# Capstone 68: Final Demo Video Script & Storyboard

**Project Question:** *How does a small trained classifier compare with a chatbot?*  
**Format:** 3-Minute Video Presentation (Screen Recording + Audio Voiceover)  
**Tools Required (Free Tier):** Screen Recorder (OBS Studio, Loom, Canva, or Clipchamp) + Microphone  

---

## 1. Video Production Overview & Storyboard

```
[0:00 - 0:30]  Scene 1: Title & The Core Dilemma (Facecam / Title Slide)
[0:30 - 1:00]  Scene 2: Google Colab & Small Model Training (<6ms)
[1:00 - 1:45]  Scene 3: Quantitative Results & The 5 Comparison Charts
[1:45 - 2:30]  Scene 4: Live Interactive Web Demo & Adversarial Testing
[2:30 - 3:00]  Scene 5: The Hybrid Solution, Responsible AI & Wrap-Up
```

---

## 2. Minute-by-Minute Script & Visual Cues

### [0:00 – 0:30] Scene 1: The Core Question & The Modern Dilemma
- **Visual on Screen:**
  Display Slide 1 from the presentation deck, followed by Slide 2 showing the two competing approaches: *Small Trained Classifier (0.2 ms)* vs *General-Purpose Chatbot (~500 ms)*.
- **Narrator Voiceover:**
  > *"When building an AI application that processes text—like sentiment analysis, support routing, or content moderation—every developer faces a major question: **Should you train a small, dedicated classifier, or just call a general-purpose chatbot like Google Gemini?**
  >
  > In this Capstone 68 project, we ran an empirical, head-to-head benchmark on movie reviews. We pitted a local scikit-learn classifier against Google Gemini 1.5 Flash to discover the exact trade-offs in accuracy, latency, linguistic nuance, and cost."*

---

### [0:30 – 1:00] Scene 2: Google Colab Pipeline & Model Training
- **Visual on Screen:**
  Screen capture of `capstone_text_classification.ipynb` running in Google Colab.
  Show the cell executing the training step: highlighting the `5.74 ms` training time and the feature extraction.
- **Narrator Voiceover:**
  > *"Here in our Google Colab notebook, we built our small classifier using sublinear TF-IDF with unigrams and bigrams, paired with a calibrated Logistic Regression model. 
  >
  > Notice the speed: the entire model trains from scratch in under six milliseconds, with an in-memory footprint of just 140 kilobytes. It requires zero cloud infrastructure, no GPU, and zero ongoing API billing."*

---

### [1:00 – 1:45] Scene 3: Benchmark Results & The 5 Comparison Charts
- **Visual on Screen:**
  Display `charts/01_metrics_comparison.png` (Bar chart), transitioning to `charts/03_latency_vs_accuracy.png` (Trade-off curve), and `charts/05_error_taxonomy.png` (Category breakdown).
- **Narrator Voiceover:**
  > *"Next, we evaluated both systems against a rigorous 100-sample benchmark specifically partitioned across six linguistic challenge types.
  >
  > Looking at our metrics chart, Gemini 1.5 Flash achieved 97% overall accuracy when optimized with few-shot Chain-of-Thought prompting, compared to 72% for our small classifier. 
  > But look at the latency trade-off curve: our small classifier responds in 0.2 milliseconds—making it over 2,500 times faster than the chatbot API call!
  >
  > When we examine our error taxonomy chart, we see exactly where the difference lies: the small model stumbles on sarcasm and modern slang like 'cooked' and 'mid', whereas Gemini's massive pretraining understands cultural nuance effortlessly."*

---

### [1:45 – 2:30] Scene 4: Live Interactive Web Demo Walkthrough
- **Visual on Screen:**
  Screen capture of the interactive demo (`demo/index.html`).
  1. Click the preset button: **"Sarcastic Flop"** (*"What a stunning achievement in testing how long an audience can endure boredom."*). Show the side-by-side card: Small Model says POSITIVE (False Positive error); Gemini says NEGATIVE (Correct!).
  2. Click the preset button: **"Gen-Z Slang"** (*"This movie absolutely slaps; the director cooked."*). Show Gemini correctly classifying it in ~430ms.
  3. Click preset button: **"Direct Praise"** (*"An enchanting, unforgettable tour de force."*). Show both models getting it right, but highlight the small model's 0.2ms vs Gemini's 450ms.
- **Narrator Voiceover:**
  > *"To demonstrate this in real time, we built an interactive comparison demo. 
  > Let's test an adversarial sarcastic review: 'What a stunning achievement in testing how long an audience can endure boredom.' 
  > Our small model sees words like 'stunning achievement' and mistakenly predicts Positive. But Gemini recognizes the sarcastic framing and correctly classifies it as Negative.
  >
  > However, on standard reviews, both models agree—except the small classifier returns the answer instantly at zero cost, while the chatbot takes nearly half a second."*

---

### [2:30 – 3:00] Scene 5: The Hybrid Architecture, Responsible AI & Conclusion
- **Visual on Screen:**
  Display Slide 10 (The Cascading Hybrid Architecture flowchart), concluding with the summary slide showing all verified repository deliverables.
- **Narrator Voiceover:**
  > *"So, what is the best industry solution? Don't pick one—use both in a **Cascading Hybrid Architecture**.
  >
  > Route incoming text through the small classifier first. If it's confident—which covers roughly 88% of standard traffic—accept the result instantly at zero cost. Only route ambiguous or low-confidence edge cases to Gemini. This delivers 95% system accuracy, sub-100 millisecond average latency, and slashes cloud API costs by nearly 90%.
  >
  > Plus, running the small model locally offers 100% data privacy and reduces carbon emissions by 99.9%.
  > All code, charts, and notebooks are available in the submission repository. Thank you for watching!"*

---

## 3. Recording Checklist for Submission
- [ ] Screen resolution set to $1920 \times 1080$ (16:9).
- [ ] Colab notebook pre-run so all output cells are cleanly visible.
- [ ] Interactive demo loaded in browser (`demo/index.html`) in full screen.
- [ ] Microphone tested with background noise reduction.
- [ ] Video exported as MP4 (under 100 MB for easy sharing).
