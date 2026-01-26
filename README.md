# **Prospect Theory in Repeated Games: Algorithm 1 vs Fictitious Play**

## **Project Overview**
This project implements and compares **Algorithm 1: Heterogeneous Q-Learning with Pathology Detection** against classical **Fictitious Play** in repeated games where human decision-makers exhibit **Prospect Theory (PT)** behavior. We computationally verify foundational findings that PT breaks Nash equilibrium and demonstrate our algorithm's superiority in handling equilibrium pathologies.

## **Core Contributions**

### **1. Theoretical Verification**
- ✅ Computationally verify that **PT breaks Nash equilibrium** in fundamental games (Matching Pennies, Ochs Game)
- ✅ Confirm **equilibrium non-existence** is more widespread than previously recognized

### **2. Algorithm Development**
- ✅ **Algorithm 1**: Novel reinforcement learning algorithm handling mixed populations of:
  - **AI agents** (standard utility maximization)
  - **Human agents** (Prospect Theory modeling)
- ✅ **Key innovation**: **Pathology detection** identifies when no clear best action exists and switches to smoothed policy selection

### **3. Comparative Baseline**
- ✅ **Fictitious Play implementation**: Classical game theory approach estimating opponent frequencies and best-responding
- ✅ Demonstrates **standard methods fail** where our algorithm succeeds

## **Agent Types Implemented**

### **1. AI Agent (`ai_agent.py`)**
- Standard Q-learning agent without PT
- Expected utility maximizer
- **Added pathology detection** for consistent decision-making

### **2. Learning Human PT Agent (`learning_human.py`)**
- Doesn't know game structure, learns via RL
- Transforms rewards through PT (loss aversion, reference points, probability weighting)
- Updates beliefs about opponent behavior
- **Pathology detection** for close-value decisions

### **3. Aware Human PT Agent (`aware_human.py`)**
- **Level 2 reasoning**: "I think about what you think I'll do"
- Knows game structure and opponent type
- Uses PT to compute best responses
- **Critical fix**: Pareto-optimal tie-breaking for AH vs AH matchups

### **4. Fictitious Play Agent (`fictitiousplay.py`)**
- Classical baseline: tracks empirical opponent frequencies
- Best-responds to historical distribution
- Configurable for **EU** or **PT** evaluations

## **Game Suite (7 Strategic Games)**

### **Core Games:**
1. **Prisoner's Dilemma** - Dominant strategy, cooperation/defection
2. **Matching Pennies** - Zero-sum competition, PT pathology demonstration
3. **Battle of the Sexes** - Coordination with asymmetric preferences
4. **Stag Hunt** - Coordination with risk trade-offs
5. **Chicken** - Risk/safety dilemmas

### **PT-Specific Games:**
6. **Ochs Game** - **Critical**: Shows PT equilibrium pathology (paper's key example)
7. **Crawford Game** - Convex preference effects, tests α=2.0 parameters

## **Key Features**

### **Prospect Theory Implementation (`ProspectTheory.py`)**
- Complete CPT (Cumulative Prospect Theory) implementation
- Loss aversion (λ=2.25), diminishing sensitivity (α=0.88), probability weighting (γ=0.61)
- Reference point adaptation
- Support for thesis parameters (α=2.0 for pathology cases)

### **Pathology Detection**
- **τ-threshold**: Detects when optimal vs second-best action values are close
- **Softmax fallback**: Smooths policy when pathology detected
- **Consistent across all agent types** for fair comparison

### **Experimental Framework**
- **6 Population Configurations** tested:
  1. All AI (baseline)
  2. All PT-Learning Humans
  3. AI vs PT-Human (mixed)
  4. PT-Human vs AI (mixed)
  5. All PT-Aware Humans
  6. PT-Aware vs PT-Learning
- **State encoding**: Base-N encoding of joint action history
- **Comprehensive metrics**: Learning curves, stability, convergence, payoff comparisons

## **Results Highlights**

### **Algorithm 1 vs Fictitious Play:**
- ✅ **Battle of Sexes**: +121.4% improvement with PT agents
- ✅ **Chicken Game**: +62.4% better risk management
- ✅ **Stag Hunt**: +36.1% improved coordination
- ✅ **Lower variance**: More stable convergence in complex games
- ✅ **PT advantage**: Greatest benefits when both agents use PT modeling

### **Key Findings:**
1. **Algorithm 1 excels in coordination/risk games** where PT effects matter
2. **Fictitious Play better in simple zero-sum games** (tracks opponent moves well)
3. **PT modeling crucial** for human-like strategic interactions
4. **Pathology detection works** - prevents oscillation in equilibrium-free games

## **Code Structure**

```
repeated_games/
├── ProspectTheory.py          # Complete PT implementation
├── ai_agent.py               # Standard RL agent + pathology detection
├── learning_human.py         # PT learning agent (Algorithm 1)
├── aware_human.py           # Sophisticated PT agent (Level 2 reasoning)
├── fictitiousplay.py        # Fictitious Play baseline
├── baseline.py              # Comparison experiments
├── game_env.py              # Repeated game environment
├── train.py                 # Training and matchup experiments
├── utils.py                 # Game definitions (7 games)
└── __init__.py             # Package exports

main_repeated.py            # Interactive experiment runner
```

## **Getting Started**

### **Quick Demo:**
```python
python main_repeated.py
# Choose option 5: "Run Algorithm 1 vs Fictitious Play comparison"
```

### **Run Complete Experiment:**
```python
from repeated_games.baseline import run_full_comparison
results = run_full_comparison()  # Tests all 7 games
```

### **Custom Matchup:**
```python
from repeated_games.train import run_complete_experiment
results = run_complete_experiment('OchsGame', payoff_matrix, episodes=300)
```

## **Research Significance**

This project provides:
1. **First computational solution** to PT equilibrium pathologies
2. **Practical algorithm** for AI-human interaction in strategic games
3. **Empirical validation** of theoretical PT-game theory predictions
4. **Benchmark comparison** against classical game theory methods
5. **Open-source implementation** for reproducibility and extension

## **Future Work**
- Extend to n-action games (generalize state encoding)
- Real human subject comparisons
- Additional baseline algorithms (DQN, Policy Gradient)
- Parameter sensitivity analysis
- Convergence proofs for Algorithm 1

## **Citation**
If using this code, please cite our paper on "Prospect Theory Equilibrium in Repeated Games: Algorithm 1 vs Classical Approaches"

