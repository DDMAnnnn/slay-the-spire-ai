# Papers behind the Spire design discussion

## 18 Nov 2026

I came to the local/global question by thinking about spam detection. I came to deck variation by analogy with image augmentation. These papers came up afterward. I want to keep track of which ideas they help me examine and what I'd need to test in Spire. These are first-pass notes from the discussion.

### How should combat and run planning talk to each other?

**Hybrid Reward Architecture for Reinforcement Learning — van Seijen et al., 2017.** In Ms. Pac-Man the authors split the reward into components and learn a value function for each. That gives me a useful comparison for my local blocks. My proposed split is by decision: a shop player might ask a combat player what tends to happen with and without a potion. I wouldn't assume that a fixed potion score plus a fixed gold score can express the value of buying access to an elite route. I'd compare those designs on shops where buying the potion changes which path the agent takes. [Paper](https://proceedings.neurips.cc/paper/2017/hash/1264a061d82a2edae1574b07249800d6-Abstract.html)

**The Option-Critic Architecture — Bacon, Harb and Precup, 2017.** An option is a policy that can persist across several actions. The paper learns both what to do during an option and when to stop. My combat/shop split already has boundaries supplied by the game. The more interesting connection would be inside a fight: if the player spends several turns setting up, can it learn when the enemy's next move makes it worth abandoning that plan? I'd want to see that failure in a replay before adding a new mechanism. [Paper](https://arxiv.org/abs/1609.05140)

**Sequential Decision Making for Improving Efficiency in Urban Environments — Varakantham, 2016.** This paper discusses planning across stages when resources are limited and later demand is uncertain. It helped put a name to something I care about in Spire: spending a potion now changes the choices I can afford later. A small test would carry the potion and HP into a second fight and see whether the player changes its first-fight decision. [Paper](https://ijcai.org/proceedings/2016/0608.pdf)

**A Simple Mixture Policy Parameterization for Improving Sample Efficiency of CVaR Optimization — Luo et al., 2024.** This work studies learning from the lower tail of returns, where relatively few training samples drive the update. It makes me want to look beyond average HP lost when comparing two elite routes. How often does each route actually kill the player? I can report that before deciding whether a risk-sensitive training objective helps my goal of winning runs. [Paper](https://cs.uwaterloo.ca/~ppoupart/publications/rl/A-Simple-Mixture-Policy-Parameterization-for-Improving-Sample-Efficiency-of-CVaR-Optimization.pdf)

### What should a combat model see and score?

**Set Transformer — Lee et al., 2019.** The authors use attention to model interactions between elements of a set. Two enemies can share an encoder while still affecting each other's target scores. I'd first swap equivalent enemy rows and check whether the scores follow the targets. Then I'd compare simple pooling with attention on fights where changing the *other* enemy changes whom I want to hit. [Paper](https://proceedings.mlr.press/v97/lee19d.html)

**Grandmaster level in StarCraft II using multi-agent reinforcement learning — Vinyals et al., 2019.** AlphaStar reads units and their properties and can select a particular unit when constructing an action. The part I might borrow is small: playing Strike on the left enemy and playing it on the right enemy are distinct choices, even though they use the same card. A two-enemy fight is enough to test whether an explicit target representation helps. [Paper](https://www.nature.com/articles/s41586-019-1724-z) · [Team's architecture walkthrough](https://deepmind.google/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii/)

**Reinforcement Learning with Dynamic Boltzmann Softmax Updates — Pan et al., 2020.** The paper studies a different way to combine next-action values and evaluates it in DQN on Atari. My MathCard DQN takes the highest legal next-action estimate. That gives me a specific question if training looks unstable: is the apparent best action repeatedly scoring higher than its realized return? I'd check that before changing the update rule. [Paper](https://ijcai.org/proceedings/2020/276)

### How can I vary decks and still tell what the model learned?

**Reinforcement Learning with Augmented Data — Laskin et al., 2020.** RAD applies data augmentation during RL training. For Spire, rearranging an unordered deck list can preserve the decision; upgrading a Strike can change whether an enemy dies. I'd use the former as an invariance check and play out the latter as a new fight. If I reorder selectable items, their action references have to move too. [Paper](https://arxiv.org/abs/2004.14990)

**Leveraging Procedural Generation to Benchmark Reinforcement Learning — Cobbe et al., 2020.** Procgen evaluates policies on generated game levels, including levels held out from training. That suggests a stronger test than dealing the same deck in a new order. I could train on two deck variants and two enemies while holding out one deck–enemy pairing. Can the player put together two things it has seen separately? [Paper](https://proceedings.mlr.press/v119/cobbe20a)

**Generalization on Text-based Games using Structured Belief Representations — Ashutosh Adhikari, 2020 (Waterloo master's thesis).** This thesis looks at structured beliefs and tests policies on unseen text games. It raises a question about my monster memory: does the history add useful information beyond the current-turn intent? I'd separately measure next-move prediction and combat performance. A player can guess the next move correctly and still play the wrong card. [Thesis](https://uwspace.uwaterloo.ca/items/7d324d73-063c-4f1a-bb1d-6b301dea2f02)

### What might carry over to a new enemy, character or difficulty?

**Contextual policy transfer in reinforcement learning domains via deep mixtures-of-experts — Gimelfarb, Sanner and Lee, 2021.** The paper asks when to reuse a source policy given the current context. This is close to my Ironclad-to-Silent idea. An Ironclad player might recognize lethal damage and still undervalue poison. I'd compare transfer by kind of decision, then test whether starting from Ironclad experience helps Silent learn faster than starting fresh. [Paper](https://proceedings.mlr.press/v161/gimelfarb21a.html)

**Knowledge Transfer for Deep Reinforcement Learning with Hierarchical Experience Replay — Yin and Pan, 2017.** Here, task-specific teachers transfer knowledge to a multitask student using replay. If I ever train specialists for individual encounters, this gives me a concrete alternative to keeping every specialist at deployment. Would one student trained from their experience beat one player trained across the same encounters at a comparable total budget? [Paper](https://cdn.aaai.org/ojs/10733/10733-13-14261-1-2-20201228.pdf)

**Paired Open-Ended Trailblazer (POET) — Wang et al., 2019.** POET evolves challenges alongside their agents and tries solutions across environments. I could borrow just the transfer check: take a frozen A20 player into lower Ascension encounters and separate enemies whose patterns change from enemies whose patterns stay the same. Overall win rate alone could hide a surprising loss of skill on one matchup. [Paper](https://arxiv.org/abs/1901.01753)

**First return, then explore — Ecoffet et al., 2021.** Go-Explore revisits promising states so that exploration can continue from places an agent rarely reaches. If my player dies early and gets almost no practice against later enemies, I could practice from saved training encounters and then evaluate fresh runs from floor one. The full-run score would tell me whether that practice actually carried over. [Paper](https://www.nature.com/articles/s41586-020-03157-9)

**RPM: Generalizable Multi-Agent Policies for Multi-Agent Reinforcement Learning — Qiu et al., 2023.** RPM keeps a ranked memory of policies to expose agents to varied partners during training. Spire is single-player, but my shop policy will effectively have a changing partner as the combat player improves. If an old shop policy stops buying useful potions, I could evaluate it with both combat versions before blaming the shop model. That's my analogy to the paper's setting. [Paper](https://personal.ntu.edu.sg/boan/papers/ICLR23_RPM.pdf)
