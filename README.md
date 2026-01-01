# HAI PREDICTIVE MODEL PROJECT
## Hospital-Acquired Infection Prevention via Optimization

**For:** Infectious Disease Physician + MITx 6.00.2x Student  
**Goal:** Build predictive models to optimize isolation bed utilization and maximize patient throughput

---

## 📋 PROJECT OVERVIEW

You're combining three powerful fields:
1. **Infectious Disease Medicine** - Understanding HAI risk factors
2. **Machine Learning** - Predicting patient outcomes
3. **Operations Research** - Optimizing hospital resources (your 6.00.2x project!)

This is the **paradox you're testing**: Can delaying some high-risk patients actually **increase** total patient throughput while **reducing** HAI rates?

---

## 📁 FILES IN THIS PACKAGE

### 📖 Documentation
- **`HAI_Predictive_Model_Design.md`** - Complete design specification
  - All 4 models explained in detail
  - Feature engineering guide
  - Training strategies
  - Integration with optimization

### 💻 Code Files
- **`hai_models_starter.py`** - Core prediction models
  - Model 1: P(HAI | Risk Factors) - Logistic Regression
  - Model 2: Length of Stay - Random Forest
  - Complete prediction pipeline
  - Working demo with synthetic data

- **`hai_visualizations.py`** - Visualization tools
  - Daily isolation bed usage charts
  - Patient timeline diagrams
  - Risk vs resource usage scatter plots
  - Optimization opportunity analysis

### 📊 Generated Visualizations
- **`isolation_usage.png`** - Shows how isolation bed demand varies over time
- **`patient_timeline.png`** - Example patient's predicted stay
- **`risk_vs_beddays.png`** - Which patients use most isolation resources
- **`optimization_opportunity.png`** - **KEY CHART** showing optimization potential

---

## 🚀 GETTING STARTED

### Step 1: Review the Design
```bash
# Read the comprehensive design document
open HAI_Predictive_Model_Design.md
```

Key sections to focus on:
- **Model 1**: How to predict P(HAI) - this is your core prediction
- **Feature Engineering**: Which risk factors matter most
- **Integrated System Output**: How all models combine

### Step 2: Run the Starter Code
```bash
# Test the models with synthetic data
python hai_models_starter.py
```

This will:
- Train Model 1 (HAI Risk) and Model 2 (LOS)
- Generate predictions for test patients
- Show you the key metric: **expected isolation bed-days**

Expected output:
```
✓ Model 1 trained: AUC-ROC: 0.77
✓ Model 2 trained: MAE: 2.8 days

Predictions:
- Low-risk patient: 2.1 isolation bed-days
- High-risk patient: 7.2 isolation bed-days
```

### Step 3: Create Visualizations
```bash
# Generate charts showing optimization opportunity
python hai_visualizations.py
```

This creates PNG files showing:
- Where isolation bed surges occur (days 4-5 in example)
- How optimization could smooth these out
- Which patients are "expensive" in isolation resources

### Step 4: Understand the Optimization Problem

**The Core Challenge:**
```
Given: 20 elective patients scheduled over 14 days
Given: 10 isolation beds available
Goal: Schedule patients to maximize throughput
Constraint: P(isolation beds > capacity) < 5%
```

**Your predictions feed the optimization:**
```python
# For each patient i, day d:
x_ipd = binary decision (0 or 1: schedule or not?)

# Constraint: Expected isolation beds on day t
E[isolation_t] = sum over all patients of:
    x_ipd * p_hai_i * overlap(patient_i, day_t)

# Must be: E[isolation_t] <= capacity
```

---

## 🎯 THE PARADOX TO EXPLORE

### Paradox 1: Delay → More Throughput
**Hypothesis:** By delaying high-risk Patient A by 2 days, you free up isolation beds to admit 3 lower-risk patients who couldn't fit before.

**Test this:** Run optimization with and without delay flexibility

### Paradox 2: Clustering > Spreading
**Hypothesis:** Scheduling 3 C.diff-risk patients on the SAME day might be better than spreading them out (cohorting allows 2 patients per room).

**Test this:** Compare clustered vs distributed schedules

### Paradox 3: Weekend = Lower HAI
**Hypothesis:** Better nurse-to-patient ratios on weekends might actually REDUCE HAI rates despite "weekend effect" concerns.

**Test this:** Stratify model by day of week

---

## 📊 KEY METRICS TO TRACK

### Model Performance
- **P(HAI) Model:**
  - AUC-ROC > 0.75 (good discrimination)
  - Calibration: predicted probabilities are accurate
  - Top risk factors: prior_cdiff, recent_abx, GI procedure

- **LOS Model:**
  - MAE < 2 days (within 2 days of actual)
  - R² > 0.5 (explains 50%+ of variance)

### Optimization Results
- **Variance Reduction:** How much smoother is optimized vs current?
- **Throughput Increase:** Can you schedule MORE patients?
- **Overflow Prevention:** Days exceeding capacity reduced to 0?
- **HAI Rate Impact:** Does optimization reduce HAI incidence?

---

## 🔬 DATA YOU'LL NEED (Real Implementation)

### From EMR (Epic, Cerner):
```sql
-- Patient risk factors
SELECT 
    patient_id,
    age,
    prior_cdiff,
    recent_abx,
    immunosuppressed,
    planned_procedure,
    los,
    developed_hai,
    hai_type,
    isolation_duration
FROM patient_admissions
WHERE admission_date > '2022-01-01'
```

### From Infection Control Database:
- NHSN reports
- C.diff surveillance
- MRSA colonization tracking
- Isolation room usage logs

### From OR Scheduling System:
- Current OR block schedule
- Surgeon-specific HAI rates
- Procedure-specific HAI risks

---

## 📈 NEXT STEPS FOR 6.00.2X

### Phase 1: Master the Predictions (Now)
- [ ] Understand all 4 models
- [ ] Run code on synthetic data
- [ ] Visualize the optimization opportunity
- [ ] Calculate expected isolation bed-days

### Phase 2: Build the MILP (Your 6.00.2x Project!)
```python
from pulp import *

# Decision variables
x = {}  # x[patient_id, day] = 0 or 1

# Objective: Minimize variance (or maximize throughput)
prob = LpProblem("HAI_Optimization", LpMinimize)

# Constraints:
# 1. Each patient scheduled at most once
# 2. Isolation beds <= capacity each day
# 3. Clinical urgency windows
# 4. OR capacity
```

### Phase 3: Test on Historical Data
- Take 6 months of historical admissions
- Predict what optimal schedule would have been
- Compare to what actually happened
- Calculate improvement metrics

### Phase 4: Prospective Pilot
- Partner with one surgical service
- Use model to schedule 1-2 weeks ahead
- Monitor outcomes
- Iterate based on feedback

---

## 🧠 CONCEPTUAL INSIGHTS

### Why This Is Hard (and Cool!)

**Stochasticity:** HAI development is probabilistic
- Patient A: 30% chance of 7-day isolation
- Patient B: 70% chance of 3-day isolation
- Who's "more expensive" in expectation? Both are 2.1 bed-days!
- But variance differs hugely

**Non-linearity:** Impact isn't additive
- 2 patients with 50% HAI risk ≠ 1 patient with 100% risk
- Because of cohorting, capacity constraints, timing

**Multi-objective:** Competing goals
- Maximize throughput
- Minimize overflow risk
- Respect clinical urgency
- Reduce HAI rates
- Keep surgeons happy

### Connection to 6.00.2x Concepts

- **Optimization:** MILP formulation (PuLP library)
- **Stochastic Models:** Monte Carlo simulation
- **Greedy Algorithms:** Baseline comparisons
- **Dynamic Programming:** Could solve smaller subproblems
- **Graph Theory:** Patient-day scheduling as bipartite matching

---

## 💡 TIPS FOR SUCCESS

### For Coding (6.00.1x → 6.00.2x Bridge)
1. **Start simple:** Get deterministic MILP working first
2. **Debug with small examples:** 5 patients, 3 days, 2 beds
3. **Visualize everything:** Charts make debugging easier
4. **Use assertions:** Check constraints are satisfied
5. **Save intermediate results:** Don't re-run expensive models

### For Machine Learning
1. **Handle class imbalance:** Use SMOTE or class weights (HAIs are rare!)
2. **Temporal validation:** Train on old data, test on recent
3. **Check calibration:** Are probabilities accurate?
4. **Feature engineering matters most:** 
   - prior_cdiff is super predictive
   - Procedure type is critical
   - Interactions (age × diabetes) help

### For Optimization
1. **Start with relaxed problem:** Allow fractional x_ipd first
2. **Add constraints incrementally:** Debug one at a time
3. **Check feasibility:** Does solution exist?
4. **Sensitivity analysis:** How do results change with capacity ±1?

---

## 📚 RELEVANT RESOURCES

### For 6.00.2x
- PuLP documentation: https://coin-or.github.io/pulp/
- MIT OpenCourseWare 6.00.2x
- "Introduction to Linear Optimization" - Bertsimas

### For Machine Learning
- Scikit-learn documentation
- "Introduction to Statistical Learning" - free PDF
- Kaggle tutorials on class imbalance

### For Healthcare Optimization
- MIT OR block scheduling paper (search: MIT flatten census)
- NHSN HAI data and benchmarks
- CDC HAI prevention guidelines

---

## 🎓 LEARNING OBJECTIVES

By completing this project, you'll master:

1. **Predictive Modeling**
   - Logistic regression for classification
   - Random forests for regression
   - Feature engineering for healthcare data
   - Model validation and calibration

2. **Optimization**
   - MILP formulation
   - Constraint satisfaction
   - Multi-objective optimization
   - Stochastic programming concepts

3. **Healthcare Analytics**
   - HAI risk stratification
   - Resource utilization modeling
   - Clinical constraint handling
   - Real-world deployment considerations

4. **Systems Thinking**
   - How predictions feed optimization
   - Balancing competing objectives
   - Stakeholder management (surgeons, IPs, administrators)

---

## 🔍 DEBUGGING CHECKLIST

If predictions seem wrong:
- [ ] Check feature scaling (Model 1 needs StandardScaler)
- [ ] Verify class imbalance handling (use class_weight='balanced')
- [ ] Confirm temporal validation (not random split)
- [ ] Review feature importance (do top features make clinical sense?)

If optimization is infeasible:
- [ ] Reduce capacity constraint temporarily
- [ ] Check admission window constraints
- [ ] Verify no patient impossible to schedule
- [ ] Try relaxed (fractional) version first

If results don't match intuition:
- [ ] Visualize the schedule
- [ ] Check predictions for outlier patients
- [ ] Verify overlap function (isolation timing)
- [ ] Print intermediate constraint values

---

## 🌟 SUCCESS CRITERIA

You'll know you've succeeded when:

✅ **Models perform well:**
- P(HAI) AUC > 0.75
- LOS MAE < 2 days
- Predictions make clinical sense

✅ **Optimization finds solutions:**
- Feasible schedules generated
- Constraints all satisfied
- Better than greedy baseline

✅ **Demonstrates paradox:**
- Strategic delays increase throughput, OR
- Clustering reduces variance, OR
- Optimization finds non-obvious improvements

✅ **Ready for deployment:**
- Code is documented
- Validation is robust
- Results are explainable to clinicians

---

## 🚦 YOUR ROADMAP

```
Week 1-2:  Master the predictions ← YOU ARE HERE
           • Understand all 4 models
           • Run starter code
           • Generate visualizations

Week 3-4:  Build MILP optimization
           • Formulate constraints
           • Implement in PuLP
           • Debug on small examples

Week 5-6:  Validate and improve
           • Test on historical data
           • Sensitivity analysis
           • Refine models

Week 7-8:  Polish and present
           • Create final visualizations
           • Write up results
           • Prepare for deployment
```

---

## 💬 QUESTIONS TO PONDER

1. **Clinical:** Should you optimize for HAI rate or isolation bed utilization? Can you do both?

2. **Mathematical:** Is minimizing variance the right objective? What about minimizing P(overflow)?

3. **Practical:** How do you get surgeon buy-in for schedule changes?

4. **Ethical:** Is it fair to delay high-risk patients? What if delay itself increases risk?

5. **Implementation:** How often should you re-train models? Re-optimize schedules?

---

## 🎉 YOU'RE READY!

You now have:
- Complete design specification
- Working starter code
- Visualizations showing the opportunity
- Clear path to optimization

**Next Action:** Start running the code and understanding the predictions. The optimization will make sense once you see what the models are telling you.

**Remember:** You're not just building a model - you're testing a paradox and potentially improving patient care at scale. This is where ID medicine meets operations research!

Good luck! 🚀

---

**Questions?** Review the design doc or explore the code comments - everything is documented for a 6.00.1x → 6.00.2x learner.
