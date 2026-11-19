# Original card translation table

Each row identifies a card by the English name and the original game ID. The catalogue below covers the card classes instantiated by the installed original game's CardLibrary. Generated cards and the three Wish choices have their own rows. Localized strings for retired designs are indexed separately at the end.

## Reading the rules

- Actions on a line run from left to right. Flags such as Innate, Ethereal, RetainSelf and ExhaustSelf describe the game's card keywords. Event handlers such as On(DrawSelf, ...) run when that event occurs. End-of-turn text on a curse in hand is a handler rather than an on-play action.
- D, B and M are card-instance damage, block and magic-number parameters. Defaults below are for a fresh card. An arrow gives the first upgraded value. Use the card instance's current upgrade and persistent mutations when binding them. Treat computed damage or block expressions as a description of the mechanic; applying bonuses again to an already adjusted UI preview would double count them.
- X is the effective X of the play. Numeric counts and expressions are arguments. Repeat(k, action) evaluates k for that repetition and performs the action k times. A random target inside Repeat is picked per repetition. Evoke(FrontOrb,k) evokes the same front orb k times and removes it once. An upgrade rule replaces the base rule; a dash means reuse the same structure with the upgraded parameters if applicable.
- Attack(target,amount) is one attack hit. Damage(target,amount) denotes a non-attack damage effect and LoseHP denotes direct HP loss. Block grants block to the player. Apply(target,status,n) adds a signed stack count. Keyword identities retain their original game semantics; their internal implementations and relic interactions are outside this card table.
- Count, Filter, Choose, Random and Top describe values or selections. Let binds a result once. Seq preserves ordering inside a handler; If selects a branch. On and During attach handlers; At schedules a future effect. NextPlayed matches a limited number of plays. Modify, SetCost and CapCost retain their duration. Combat card collections follow the normal game pile rules; Hand in on-play instructions excludes the card being played.
- A power's handlers last for the combat unless given another duration. Fatal uses the game's Fatal condition. Cost X and cost — denote X-cost and normally unplayable cards. Cost is the unmodified energy cost. The table describes card behavior and is not a simulator or an action policy.

## How the vocabulary changed during the pass

| Case | Shared representation | Distinction that matters |
|---|---|---|
| Strike | `Attack(Target,D)` | One attack hit on the selected enemy |
| Twin Strike, Pummel, Skewer | `Repeat(k,Attack(Target,D))` | k can be a literal, a card parameter, or X |
| Barrage, Flechettes, Finisher | The same Repeat with a count expression | Occupied orb slots, Skills in hand, and prior Attacks have different scopes |
| Proposed relic-count attack | `Repeat(Count(PlayerRelics),Attack(Target,10))` | A new combination, with no new attack or repeat type |
| Body Slam versus Barrage | A count or state value can fill either argument | One large hit differs from many small hits |
| Sword Boomerang versus Whirlwind | Target selection stays inside the repeated attack | Random draws can select a different enemy per hit; an area hit affects each enemy |
| Defend versus Spirit Shield | `Block(B)` or `Block(Mul(M,Count(OtherHandCards)))` | Spirit Shield grants block once; its quantity depends on hand size |
| Fiend Fire | Bind the number actually exhausted, then repeat | Reading hand size after exhausting the hand would lose the count |

## Ironclad (75)

| Card | Game ID | Cost | Parameters (base → first upgrade) | Base rule | Changed rule on upgrade |
|---|---|---|---|---|---|
| Anger | `Anger` | 0 | D=6→8 | `Attack(Target,D); Create(Copy(Self),DiscardPile,1)` | — |
| Armaments | `Armaments` | 1 | B=5 | `Block(B); Upgrade(Choose(Hand,1),Combat)` | `Block(B); Upgrade(Hand,Combat)` |
| Barricade | `Barricade` | 3→2 | — | `Prevent(ClearBlockAtTurnStart,Combat)` | — |
| Bash | `Bash` | 2 | D=8→10, M=2→3 | `Attack(Target,D); Apply(Target,Vulnerable,M)` | — |
| Battle Trance | `Battle Trance` | 0 | M=3→4 | `Draw(M); Prevent(DrawCard,Turn)` | — |
| Berserk | `Berserk` | 0 | M=2→1 | `Apply(Player,Vulnerable,M); On(TurnStart,Energy(1))` | — |
| Blood for Blood | `Blood for Blood` | 4→3 | D=18→22 | `Modify(SelfCost,Subtract(CountEvents(PlayerLostHP,Combat)),Combat); Attack(Target,D)` | — |
| Bloodletting | `Bloodletting` | 0 | — | `LoseHP(Player,3); Energy(2)` | `LoseHP(Player,3); Energy(3)` |
| Bludgeon | `Bludgeon` | 3 | D=32→42 | `Attack(Target,D)` | — |
| Body Slam | `Body Slam` | 1→0 | — | `Attack(Target,PlayerBlock)` | — |
| Brutality | `Brutality` | 0 | — | `On(TurnStart,Seq(LoseHP(Player,1),Draw(1)))` | `Innate; On(TurnStart,Seq(LoseHP(Player,1),Draw(1)))` |
| Burning Pact | `Burning Pact` | 1 | M=2→3 | `Exhaust(Choose(Hand,1)); Draw(M)` | — |
| Carnage | `Carnage` | 2 | D=20→28 | `Ethereal; Attack(Target,D)` | — |
| Clash | `Clash` | 0 | D=14→18 | `Require(AllMatch(Hand,Attack)); Attack(Target,D)` | — |
| Cleave | `Cleave` | 1 | D=8→11 | `Attack(AllEnemies,D)` | — |
| Clothesline | `Clothesline` | 2 | D=12→14, M=2→3 | `Attack(Target,D); Apply(Target,Weak,M)` | — |
| Combust | `Combust` | 1 | M=5→7 | `On(TurnEnd,Seq(LoseHP(Player,1),Damage(AllEnemies,M)))` | — |
| Corruption | `Corruption` | 3→2 | — | `SetCost(Skills,0,Combat); On(PlaySkill,Exhaust(PlayedCard))` | — |
| Dark Embrace | `Dark Embrace` | 2→1 | — | `On(ExhaustCard,Draw(1))` | — |
| Defend | `Defend_R` | 1 | B=5→8 | `Block(B)` | — |
| Demon Form | `Demon Form` | 3 | M=2→3 | `On(TurnStart,Apply(Player,Strength,M))` | — |
| Disarm | `Disarm` | 1 | M=2→3 | `Apply(Target,Strength,Neg(M)); ExhaustSelf` | — |
| Double Tap | `Double Tap` | 1 | M=1→2 | `NextPlayed(Attack,1,Turn,Replay(1))` | `NextPlayed(Attack,M,Turn,Replay(1))` |
| Dropkick | `Dropkick` | 1 | D=5→8 | `Attack(Target,D); If(Has(Target,Vulnerable),Seq(Energy(1),Draw(1)))` | — |
| Dual Wield | `Dual Wield` | 1 | — | `Create(Copy(Choose(Filter(Hand,AttackOrPower),1)),Hand,1)` | `Create(Copy(Choose(Filter(Hand,AttackOrPower),1)),Hand,2)` |
| Entrench | `Entrench` | 2→1 | — | `Block(PlayerBlock)` | — |
| Evolve | `Evolve` | 1 | M=1→2 | `On(DrawStatus,Draw(M))` | — |
| Exhume | `Exhume` | 1→0 | — | `Move(Choose(ExhaustPileExceptExhume,1),Hand); ExhaustSelf` | — |
| Feed | `Feed` | 1 | D=10→12, M=3→4 | `Attack(Target,D); If(Fatal,GainMaxHP(M)); ExhaustSelf` | — |
| Feel No Pain | `Feel No Pain` | 1 | M=3→4 | `On(ExhaustCard,Block(M))` | — |
| Fiend Fire | `Fiend Fire` | 2 | D=7→10 | `Let(n,Count(Exhaust(Hand)),Repeat(n,Attack(Target,D))); ExhaustSelf` | — |
| Fire Breathing | `Fire Breathing` | 1 | M=6→10 | `On(DrawStatusOrCurse,Damage(AllEnemies,M))` | — |
| Flame Barrier | `Flame Barrier` | 2 | B=12→16, M=4→6 | `Block(B); During(Turn,On(Attacked,Damage(Attacker,M)))` | — |
| Flex | `Flex` | 0 | M=2→4 | `TempApply(Player,Strength,M,Turn)` | — |
| Ghostly Armor | `Ghostly Armor` | 1 | B=10→13 | `Ethereal; Block(B)` | — |
| Havoc | `Havoc` | 1→0 | — | `Play(Top(DrawPile),RandomEnemy); Exhaust(PlayedCard)` | — |
| Headbutt | `Headbutt` | 1 | D=9→12 | `Attack(Target,D); Move(Choose(DiscardPile,1),DrawPileTop)` | — |
| Heavy Blade | `Heavy Blade` | 2 | D=14, M=3→5 | `Attack(Target,WithStrengthMultiplier(D,M))` | — |
| Hemokinesis | `Hemokinesis` | 1 | D=15→20, M=2 | `LoseHP(Player,M); Attack(Target,D)` | — |
| Immolate | `Immolate` | 2 | D=21→28 | `Attack(AllEnemies,D); Create(Burn,DiscardPile,1)` | — |
| Impervious | `Impervious` | 2 | B=30→40 | `Block(B); ExhaustSelf` | — |
| Infernal Blade | `Infernal Blade` | 1→0 | — | `Create(RandomAttack,Hand,1,CostForTurn(0)); ExhaustSelf` | — |
| Inflame | `Inflame` | 1 | M=2→3 | `Apply(Player,Strength,M)` | — |
| Intimidate | `Intimidate` | 0 | M=1→2 | `Apply(AllEnemies,Weak,M); ExhaustSelf` | — |
| Iron Wave | `Iron Wave` | 1 | D=5→7, B=5→7 | `Block(B); Attack(Target,D)` | — |
| Juggernaut | `Juggernaut` | 2 | M=5→7 | `On(GainBlock,Damage(RandomEnemy,M))` | — |
| Limit Break | `Limit Break` | 1 | — | `ScaleStatus(Player,Strength,2); ExhaustSelf` | `ScaleStatus(Player,Strength,2)` |
| Metallicize | `Metallicize` | 1 | M=3→4 | `On(TurnEnd,Block(M))` | — |
| Offering | `Offering` | 0 | M=3→5 | `LoseHP(Player,6); Energy(2); Draw(M); ExhaustSelf` | — |
| Perfected Strike | `Perfected Strike` | 2 | D=6, M=2→3 | `Attack(Target,WithCardBonus(D,Mul(M,Count(StrikeTaggedCombatCards))))` | — |
| Pommel Strike | `Pommel Strike` | 1 | D=9→10, M=1→2 | `Attack(Target,D); Draw(M)` | — |
| Power Through | `Power Through` | 1 | B=15→20 | `Create(Wound,Hand,2); Block(B)` | — |
| Pummel | `Pummel` | 1 | D=2, M=4→5 | `Repeat(M,Attack(Target,D)); ExhaustSelf` | — |
| Rage | `Rage` | 0 | M=3→5 | `During(Turn,On(PlayAttack,Block(M)))` | — |
| Rampage | `Rampage` | 1 | D=8, M=5→8 | `Attack(Target,D); Modify(DamageOf(Self),Add(M),Combat)` | — |
| Reaper | `Reaper` | 2 | D=4→5 | `Attack(AllEnemies,D); Heal(LastUnblockedDamageTotal); ExhaustSelf` | — |
| Reckless Charge | `Reckless Charge` | 0 | D=7→10 | `Attack(Target,D); Create(Dazed,DrawPileRandom,1)` | — |
| Rupture | `Rupture` | 1 | M=1→2 | `On(LoseHPFromCard,Apply(Player,Strength,M))` | — |
| Searing Blow | `Searing Blow` | 2 | D=12→16 | `Attack(Target,D); On(UpgradeSelf,Modify(DamageOf(Self),Add(Add(4,PriorUpgrades)),Permanent)); AllowRepeatedUpgrades` | — |
| Second Wind | `Second Wind` | 1 | B=5→7 | `Let(n,Count(Exhaust(Filter(Hand,NonAttack))),Repeat(n,Block(B)))` | — |
| Seeing Red | `Seeing Red` | 1→0 | — | `Energy(2); ExhaustSelf` | — |
| Sentinel | `Sentinel` | 1 | B=5→8 | `Block(B); On(ExhaustSelf,Energy(2))` | `Block(B); On(ExhaustSelf,Energy(3))` |
| Sever Soul | `Sever Soul` | 2 | D=16→22 | `Exhaust(Filter(Hand,NonAttack)); Attack(Target,D)` | — |
| Shockwave | `Shockwave` | 2 | M=3→5 | `Apply(AllEnemies,Weak,M); Apply(AllEnemies,Vulnerable,M); ExhaustSelf` | — |
| Shrug It Off | `Shrug It Off` | 1 | B=8→11 | `Block(B); Draw(1)` | — |
| Spot Weakness | `Spot Weakness` | 1 | M=3→4 | `If(IntendsAttack(Target),Apply(Player,Strength,M))` | — |
| Strike | `Strike_R` | 1 | D=6→9 | `Attack(Target,D)` | — |
| Sword Boomerang | `Sword Boomerang` | 1 | D=3, M=3→4 | `Repeat(M,Attack(RandomEnemy,D))` | — |
| Thunderclap | `Thunderclap` | 1 | D=4→7 | `Attack(AllEnemies,D); Apply(AllEnemies,Vulnerable,1)` | — |
| True Grit | `True Grit` | 1 | B=7→9 | `Block(B); Exhaust(Random(Hand,1))` | `Block(B); Exhaust(Choose(Hand,1))` |
| Twin Strike | `Twin Strike` | 1 | D=5→7 | `Repeat(2,Attack(Target,D))` | — |
| Uppercut | `Uppercut` | 2 | D=13, M=1→2 | `Attack(Target,D); Apply(Target,Weak,M); Apply(Target,Vulnerable,M)` | — |
| Warcry | `Warcry` | 0 | M=1→2 | `Draw(M); Move(Choose(Hand,1),DrawPileTop); ExhaustSelf` | — |
| Whirlwind | `Whirlwind` | X | D=5→8 | `Repeat(X,Attack(AllEnemies,D))` | — |
| Wild Strike | `Wild Strike` | 1 | D=12→17 | `Attack(Target,D); Create(Wound,DrawPileRandom,1)` | — |

## Silent (75)

| Card | Game ID | Cost | Parameters (base → first upgrade) | Base rule | Changed rule on upgrade |
|---|---|---|---|---|---|
| A Thousand Cuts | `A Thousand Cuts` | 2 | M=1→2 | `On(PlayCard, Damage(AllEnemies,M))` | — |
| Accuracy | `Accuracy` | 1 | M=4→6 | `Modify(DamageOf(Shiv),Add(M),Combat)` | — |
| Acrobatics | `Acrobatics` | 1 | M=3→4 | `Draw(M); Discard(Choose(Hand,1))` | — |
| Adrenaline | `Adrenaline` | 0 | — | `Energy(1); Draw(2); ExhaustSelf` | `Energy(2); Draw(2); ExhaustSelf` |
| After Image | `After Image` | 1 | — | `On(PlayCard,Block(1))` | `Innate; On(PlayCard,Block(1))` |
| Alchemize | `Venomology` | 1→0 | — | `Obtain(RandomPotion,1); ExhaustSelf` | — |
| All-Out Attack | `All Out Attack` | 1 | D=10→14 | `Attack(AllEnemies,D); Discard(Random(Hand,1))` | — |
| Backflip | `Backflip` | 1 | B=5→8 | `Block(B); Draw(2)` | — |
| Backstab | `Backstab` | 0 | D=11→15 | `Innate; Attack(Target,D); ExhaustSelf` | — |
| Bane | `Bane` | 1 | D=7→10 | `Attack(Target,D); If(Has(Target,Poison),Attack(Target,D))` | — |
| Blade Dance | `Blade Dance` | 1 | M=3→4 | `Create(Shiv,Hand,M)` | — |
| Blur | `Blur` | 1 | B=5→8 | `Block(B); Prevent(ClearBlockAtTurnStart,NextTurnStart)` | — |
| Bouncing Flask | `Bouncing Flask` | 2 | M=3→4 | `Repeat(M,Apply(RandomEnemy,Poison,3))` | — |
| Bullet Time | `Bullet Time` | 3→2 | — | `Prevent(DrawCard,Turn); SetCost(Hand,0,Turn)` | — |
| Burst | `Burst` | 1 | M=1→2 | `NextPlayed(Skill,1,Turn,Replay(1))` | `NextPlayed(Skill,M,Turn,Replay(1))` |
| Calculated Gamble | `Calculated Gamble` | 0 | — | `Let(n,Count(Hand),Seq(Discard(Hand),Draw(n))); ExhaustSelf` | `Let(n,Count(Hand),Seq(Discard(Hand),Draw(n)))` |
| Caltrops | `Caltrops` | 1 | M=3→5 | `On(Attacked,Damage(Attacker,M))` | — |
| Catalyst | `Catalyst` | 1 | — | `ScaleStatus(Target,Poison,2); ExhaustSelf` | `ScaleStatus(Target,Poison,3); ExhaustSelf` |
| Choke | `Choke` | 2 | D=12, M=3→5 | `Attack(Target,D); During(Turn,On(PlayCard,LoseHP(Target,M)))` | — |
| Cloak and Dagger | `Cloak And Dagger` | 1 | B=6, M=1→2 | `Block(B); Create(Shiv,Hand,M)` | — |
| Concentrate | `Concentrate` | 0 | M=3→2 | `Discard(Choose(Hand,M)); Energy(2)` | — |
| Corpse Explosion | `Corpse Explosion` | 2 | M=6→9 | `Apply(Target,Poison,M); OnDeathOf(Target,Damage(AllEnemies,MaxHP(Target)))` | — |
| Crippling Cloud | `Crippling Poison` | 2 | M=4→7 | `Apply(AllEnemies,Poison,M); Apply(AllEnemies,Weak,2); ExhaustSelf` | — |
| Dagger Spray | `Dagger Spray` | 1 | D=4→6 | `Repeat(2,Attack(AllEnemies,D))` | — |
| Dagger Throw | `Dagger Throw` | 1 | D=9→12 | `Attack(Target,D); Draw(1); Discard(Choose(Hand,1))` | — |
| Dash | `Dash` | 2 | D=10→13, B=10→13 | `Block(B); Attack(Target,D)` | — |
| Deadly Poison | `Deadly Poison` | 1 | M=5→7 | `Apply(Target,Poison,M)` | — |
| Defend | `Defend_G` | 1 | B=5→8 | `Block(B)` | — |
| Deflect | `Deflect` | 0 | B=4→7 | `Block(B)` | — |
| Die Die Die | `Die Die Die` | 1 | D=13→17 | `Attack(AllEnemies,D); ExhaustSelf` | — |
| Distraction | `Distraction` | 1→0 | — | `Create(RandomSkill,Hand,1,CostForTurn(0)); ExhaustSelf` | — |
| Dodge and Roll | `Dodge and Roll` | 1 | B=4→6 | `Block(B); At(NextTurnStart,Block(B))` | — |
| Doppelganger | `Doppelganger` | X | — | `At(NextTurnStart,Seq(Draw(X),Energy(X))); ExhaustSelf` | `At(NextTurnStart,Seq(Draw(Add(X,1)),Energy(Add(X,1)))); ExhaustSelf` |
| Endless Agony | `Endless Agony` | 0 | D=4→6 | `Attack(Target,D); On(DrawSelf,Create(Copy(Self),Hand,1)); ExhaustSelf` | — |
| Envenom | `Envenom` | 2→1 | — | `On(UnblockedAttackHit,Apply(HitTarget,Poison,1))` | — |
| Escape Plan | `Escape Plan` | 0 | B=3→5 | `Draw(1); If(Is(LastDrawn,Skill),Block(B))` | — |
| Eviscerate | `Eviscerate` | 3 | D=7→9 | `Modify(SelfCost,Subtract(CountEvents(DiscardCard,Turn)),Turn); Repeat(3,Attack(Target,D))` | — |
| Expertise | `Expertise` | 1 | M=6→7 | `Draw(Max(0,Subtract(M,Count(Hand))))` | — |
| Finisher | `Finisher` | 1 | D=6→8 | `Repeat(CountEvents(PriorAttack,Turn),Attack(Target,D))` | — |
| Flechettes | `Flechettes` | 1 | D=4→6 | `Repeat(Count(Filter(Hand,Skill)),Attack(Target,D))` | — |
| Flying Knee | `Flying Knee` | 1 | D=8→11 | `Attack(Target,D); At(NextTurnStart,Energy(1))` | — |
| Footwork | `Footwork` | 1 | M=2→3 | `Apply(Player,Dexterity,M)` | — |
| Glass Knife | `Glass Knife` | 1 | D=8→12 | `Repeat(2,Attack(Target,D)); Modify(DamageOf(Self),Subtract(2),Combat)` | — |
| Grand Finale | `Grand Finale` | 0 | D=50→60 | `Require(Empty(DrawPile)); Attack(AllEnemies,D)` | — |
| Heel Hook | `Heel Hook` | 1 | D=5→8 | `Attack(Target,D); If(Has(Target,Weak),Seq(Energy(1),Draw(1)))` | — |
| Infinite Blades | `Infinite Blades` | 1 | — | `On(TurnStart,Create(Shiv,Hand,1))` | `Innate; On(TurnStart,Create(Shiv,Hand,1))` |
| Leg Sweep | `Leg Sweep` | 2 | B=11→14, M=2→3 | `Apply(Target,Weak,M); Block(B)` | — |
| Malaise | `Malaise` | X | — | `Apply(Target,Strength,Neg(X)); Apply(Target,Weak,X); ExhaustSelf` | `Apply(Target,Strength,Neg(Add(X,1))); Apply(Target,Weak,Add(X,1)); ExhaustSelf` |
| Masterful Stab | `Masterful Stab` | 0 | D=12→16 | `Modify(SelfCost,Add(CountEvents(PlayerLostHP,Combat)),Combat); Attack(Target,D)` | — |
| Neutralize | `Neutralize` | 0 | D=3→4, M=1→2 | `Attack(Target,D); Apply(Target,Weak,M)` | — |
| Nightmare | `Night Terror` | 3→2 | M=3 | `Let(c,Choose(Hand,1),At(NextTurnStart,Create(Copy(c),Hand,M))); ExhaustSelf` | — |
| Noxious Fumes | `Noxious Fumes` | 1 | M=2→3 | `On(TurnStart,Apply(AllEnemies,Poison,M))` | — |
| Outmaneuver | `Outmaneuver` | 1 | — | `At(NextTurnStart,Energy(2))` | `At(NextTurnStart,Energy(3))` |
| Phantasmal Killer | `Phantasmal Killer` | 1→0 | — | `At(NextTurnStart,During(Turn,MultiplyAttackDamage(2)))` | — |
| Piercing Wail | `PiercingWail` | 1 | M=6→8 | `TempApply(AllEnemies,Strength,Neg(M),Turn); ExhaustSelf` | — |
| Poisoned Stab | `Poisoned Stab` | 1 | D=6→8, M=3→4 | `Attack(Target,D); Apply(Target,Poison,M)` | — |
| Predator | `Predator` | 2 | D=15→20 | `Attack(Target,D); At(NextTurnStart,Draw(2))` | — |
| Prepared | `Prepared` | 0 | M=1→2 | `Draw(M); Discard(Choose(Hand,M))` | — |
| Quick Slash | `Quick Slash` | 1 | D=8→12 | `Attack(Target,D); Draw(1)` | — |
| Reflex | `Reflex` | — | M=2→3 | `Unplayable; On(DiscardSelfFromHand,Draw(M))` | — |
| Riddle with Holes | `Riddle With Holes` | 2 | D=3→4 | `Repeat(5,Attack(Target,D))` | — |
| Setup | `Setup` | 1→0 | — | `Let(c,Choose(Hand,1),Seq(Move(c,DrawPileTop),SetCost(c,0,UntilPlayed)))` | — |
| Skewer | `Skewer` | X | D=7→10 | `Repeat(X,Attack(Target,D))` | — |
| Slice | `Slice` | 0 | D=6→9 | `Attack(Target,D)` | — |
| Sneaky Strike | `Underhanded Strike` | 2 | D=12→16 | `Attack(Target,D); If(Positive(CountEvents(DiscardCard,Turn)),Energy(2))` | — |
| Storm of Steel | `Storm of Steel` | 1 | — | `Let(n,Count(Discard(Hand)),Create(Shiv,Hand,n))` | `Let(n,Count(Discard(Hand)),Create(Upgraded(Shiv),Hand,n))` |
| Strike | `Strike_G` | 1 | D=6→9 | `Attack(Target,D)` | — |
| Sucker Punch | `Sucker Punch` | 1 | D=7→9, M=1→2 | `Attack(Target,D); Apply(Target,Weak,M)` | — |
| Survivor | `Survivor` | 1 | B=8→11 | `Block(B); Discard(Choose(Hand,1))` | — |
| Tactician | `Tactician` | — | — | `Unplayable; On(DiscardSelfFromHand,Energy(1))` | `Unplayable; On(DiscardSelfFromHand,Energy(2))` |
| Terror | `Terror` | 1→0 | — | `Apply(Target,Vulnerable,99); ExhaustSelf` | — |
| Tools of the Trade | `Tools of the Trade` | 1→0 | — | `On(TurnStart,Seq(Draw(1),Discard(Choose(Hand,1))))` | — |
| Unload | `Unload` | 1 | D=14→18 | `Attack(Target,D); Discard(Filter(Hand,NonAttack))` | — |
| Well-Laid Plans | `Well Laid Plans` | 1 | M=1→2 | `On(TurnEnd,Retain(ChooseUpTo(Hand,M),Turn))` | — |
| Wraith Form | `Wraith Form v2` | 3 | M=2→3 | `Apply(Player,Intangible,M); On(TurnEnd,Apply(Player,Dexterity,-1))` | — |

## Defect (75)

| Card | Game ID | Cost | Parameters (base → first upgrade) | Base rule | Changed rule on upgrade |
|---|---|---|---|---|---|
| Aggregate | `Aggregate` | 1 | M=4→3 | `Energy(FloorDiv(Count(DrawPile),M))` | — |
| All for One | `All For One` | 2 | D=10→14 | `Attack(Target,D); Move(Filter(DiscardPile,Cost(0)),Hand)` | — |
| Amplify | `Amplify` | 1 | M=1→2 | `NextPlayed(Power,1,Turn,Replay(1))` | `NextPlayed(Power,M,Turn,Replay(1))` |
| Auto-Shields | `Auto Shields` | 1 | B=11→15 | `If(Equal(PlayerBlock,0),Block(B))` | — |
| Ball Lightning | `Ball Lightning` | 1 | D=7→10, M=1 | `Attack(Target,D); Channel(Lightning,M)` | — |
| Barrage | `Barrage` | 1 | D=4→6 | `Repeat(Count(OccupiedOrbSlots),Attack(Target,D))` | — |
| Beam Cell | `Beam Cell` | 0 | D=3→4, M=1→2 | `Attack(Target,D); Apply(Target,Vulnerable,M)` | — |
| Biased Cognition | `Biased Cognition` | 1 | M=4→5 | `Apply(Player,Focus,M); On(TurnStart,Apply(Player,Focus,-1))` | — |
| Blizzard | `Blizzard` | 1 | M=2→3 | `Attack(AllEnemies,Mul(M,CountEvents(ChannelFrost,Combat)))` | — |
| Boot Sequence | `BootSequence` | 0 | B=10→13 | `Innate; Block(B); ExhaustSelf` | — |
| Buffer | `Buffer` | 2 | M=1→2 | `PreventNext(PlayerLoseHP,1)` | `PreventNext(PlayerLoseHP,M)` |
| Bullseye | `Lockon` | 1 | D=8→11, M=2→3 | `Attack(Target,D); Apply(Target,LockOn,M)` | — |
| Capacitor | `Capacitor` | 1 | M=2→3 | `AddOrbSlots(M)` | — |
| Chaos | `Chaos` | 1 | M=1→2 | `Repeat(M,Channel(RandomOrbType,1))` | — |
| Charge Battery | `Conserve Battery` | 1 | B=7→10 | `Block(B); At(NextTurnStart,Energy(1))` | — |
| Chill | `Chill` | 0 | M=1 | `Repeat(Count(LivingEnemies),Channel(Frost,M)); ExhaustSelf` | `Innate; Repeat(Count(LivingEnemies),Channel(Frost,M)); ExhaustSelf` |
| Claw | `Gash` | 0 | D=3→5, M=2 | `Attack(Target,D); Modify(DamageOf(AllClaws),Add(M),Combat)` | — |
| Cold Snap | `Cold Snap` | 1 | D=6→9, M=1 | `Attack(Target,D); Channel(Frost,M)` | — |
| Compile Driver | `Compile Driver` | 1 | D=7→10, M=1 | `Attack(Target,D); Draw(Mul(M,Count(DistinctOrbTypes)))` | — |
| Consume | `Consume` | 2 | M=2→3 | `Apply(Player,Focus,M); AddOrbSlots(-1)` | — |
| Coolheaded | `Coolheaded` | 1 | M=1→2 | `Channel(Frost,1); Draw(M)` | — |
| Core Surge | `Core Surge` | 1 | D=11→15, M=1 | `Attack(Target,D); Apply(Player,Artifact,M); ExhaustSelf` | — |
| Creative AI | `Creative AI` | 3→2 | — | `On(TurnStart,Create(RandomPower,Hand,1))` | — |
| Darkness | `Darkness` | 1 | M=1 | `Channel(Dark,M)` | `Channel(Dark,M); ForEach(Filter(OrbSlots,Dark),TriggerPassive(Item))` |
| Defend | `Defend_B` | 1 | B=5→8 | `Block(B)` | — |
| Defragment | `Defragment` | 1 | M=1→2 | `Apply(Player,Focus,M)` | — |
| Doom and Gloom | `Doom and Gloom` | 2 | D=10→14, M=1 | `Attack(AllEnemies,D); Channel(Dark,M)` | — |
| Double Energy | `Double Energy` | 1→0 | — | `Energy(CurrentEnergy); ExhaustSelf` | `Energy(CurrentEnergy)` |
| Dualcast | `Dualcast` | 1→0 | — | `Evoke(FrontOrb,2)` | — |
| Echo Form | `Echo Form` | 3 | — | `Ethereal; On(FirstCardEachTurn,Replay(1))` | `On(FirstCardEachTurn,Replay(1))` |
| Electrodynamics | `Electrodynamics` | 2 | M=2→3 | `Redirect(Lightning,AllEnemies,Combat); Channel(Lightning,M)` | — |
| Equilibrium | `Undo` | 2 | B=13→16 | `Block(B); Retain(Hand,Turn)` | — |
| Fission | `Fission` | 0 | M=1 | `Let(n,Count(Remove(OccupiedOrbSlots)),Repeat(n,Seq(Energy(1),Draw(M)))); ExhaustSelf` | `Let(n,Count(EvokeAll),Repeat(n,Seq(Energy(1),Draw(M)))); ExhaustSelf` |
| Force Field | `Force Field` | 4 | B=12→16 | `Modify(SelfCost,Subtract(CountEvents(PlayPower,Combat)),Combat); Block(B)` | — |
| FTL | `FTL` | 0 | D=5→6, M=3→4 | `Attack(Target,D); If(Less(CardsPlayedBeforeSelfThisTurn,M),Draw(1))` | — |
| Fusion | `Fusion` | 2→1 | M=1 | `Channel(Plasma,M)` | — |
| Genetic Algorithm | `Genetic Algorithm` | 1 | B=1, M=2→3 | `Block(B); Modify(BlockOf(Self),Add(M),Permanent); ExhaustSelf` | — |
| Glacier | `Glacier` | 2 | B=7→10, M=2 | `Block(B); Channel(Frost,M)` | — |
| Go for the Eyes | `Go for the Eyes` | 0 | D=3→4, M=1→2 | `Attack(Target,D); If(IntendsAttack(Target),Apply(Target,Weak,M))` | — |
| Heatsinks | `Heatsinks` | 1 | M=1→2 | `On(PlayPower,Draw(M))` | — |
| Hello World | `Hello World` | 1 | — | `On(TurnStart,Create(RandomCommon,Hand,1))` | `Innate; On(TurnStart,Create(RandomCommon,Hand,1))` |
| Hologram | `Hologram` | 1 | B=3→5 | `Block(B); Move(Choose(DiscardPile,1),Hand); ExhaustSelf` | `Block(B); Move(Choose(DiscardPile,1),Hand)` |
| Hyperbeam | `Hyperbeam` | 2 | D=26→34, M=3 | `Attack(AllEnemies,D); Apply(Player,Focus,Neg(M))` | — |
| Leap | `Leap` | 1 | B=9→12 | `Block(B)` | — |
| Loop | `Loop` | 1 | M=1→2 | `On(TurnStart,TriggerPassive(FrontOrb))` | `On(TurnStart,Repeat(M,TriggerPassive(FrontOrb)))` |
| Machine Learning | `Machine Learning` | 1 | M=1 | `On(TurnStart,Draw(M))` | `Innate; On(TurnStart,Draw(M))` |
| Melter | `Melter` | 1 | D=10→14 | `RemoveBlock(Target); Attack(Target,D)` | — |
| Meteor Strike | `Meteor Strike` | 5 | D=24→30, M=3 | `Attack(Target,D); Channel(Plasma,M)` | — |
| Multi-Cast | `Multi-Cast` | X | — | `Evoke(FrontOrb,X)` | `Evoke(FrontOrb,Add(X,1))` |
| Overclock | `Steam Power` | 0 | M=2→3 | `Draw(M); Create(Burn,DiscardPile,1)` | — |
| Rainbow | `Rainbow` | 2 | — | `Channel(Lightning,1); Channel(Frost,1); Channel(Dark,1); ExhaustSelf` | `Channel(Lightning,1); Channel(Frost,1); Channel(Dark,1)` |
| Reboot | `Reboot` | 0 | M=4→6 | `ShuffleInto(HandAndDiscard,DrawPile); Draw(M); ExhaustSelf` | — |
| Rebound | `Rebound` | 1 | D=9→12 | `Attack(Target,D); NextPlayed(AnyCard,1,Turn,Move(PlayedCard,DrawPileTop))` | — |
| Recursion | `Redo` | 1→0 | — | `Let(t,TypeOf(FrontOrb),Seq(Evoke(FrontOrb,1),Channel(t,1)))` | — |
| Recycle | `Recycle` | 1→0 | — | `Let(c,Choose(Hand,1),Seq(Exhaust(c),Energy(Cost(c))))` | — |
| Reinforced Body | `Reinforced Body` | X | B=7→9 | `Repeat(X,Block(B))` | — |
| Reprogram | `Reprogram` | 1 | M=1→2 | `Apply(Player,Focus,Neg(M)); Apply(Player,Strength,M); Apply(Player,Dexterity,M)` | — |
| Rip and Tear | `Rip and Tear` | 1 | D=7→9 | `Repeat(2,Attack(RandomEnemy,D))` | — |
| Scrape | `Scrape` | 1 | D=7→10, M=4→5 | `Attack(Target,D); Let(cards,Draw(M),Discard(Filter(cards,NonzeroCost)))` | — |
| Seek | `Seek` | 0 | M=1→2 | `Move(Choose(DrawPile,M),Hand); ExhaustSelf` | — |
| Self Repair | `Self Repair` | 1 | M=7→10 | `On(CombatEnd,Heal(M))` | — |
| Skim | `Skim` | 1 | M=3→4 | `Draw(M)` | — |
| Stack | `Stack` | 1 | — | `Block(Count(DiscardPile))` | `Block(Add(Count(DiscardPile),3))` |
| Static Discharge | `Static Discharge` | 1 | M=1→2 | `On(ReceiveUnblockedAttackDamage,Channel(Lightning,M))` | — |
| Steam Barrier | `Steam` | 0 | B=6→8 | `Block(B); Modify(BlockOf(Self),Subtract(1),Combat)` | — |
| Storm | `Storm` | 1 | — | `On(PlayPower,Channel(Lightning,1))` | `Innate; On(PlayPower,Channel(Lightning,1))` |
| Streamline | `Streamline` | 2 | D=15→20, M=1 | `Attack(Target,D); Modify(SelfCost,Subtract(M),Combat)` | — |
| Strike | `Strike_B` | 1 | D=6→9 | `Attack(Target,D)` | — |
| Sunder | `Sunder` | 3 | D=24→32 | `Attack(Target,D); If(KilledTarget,Energy(3))` | — |
| Sweeping Beam | `Sweeping Beam` | 1 | D=6→9, M=1 | `Attack(AllEnemies,D); Draw(M)` | — |
| Tempest | `Tempest` | X | — | `Channel(Lightning,X); ExhaustSelf` | `Channel(Lightning,Add(X,1)); ExhaustSelf` |
| Thunder Strike | `Thunder Strike` | 3 | D=7→9 | `Repeat(CountEvents(ChannelLightning,Combat),Attack(RandomEnemy,D))` | — |
| TURBO | `Turbo` | 0 | — | `Energy(2); Create(Void,DiscardPile,1)` | `Energy(3); Create(Void,DiscardPile,1)` |
| White Noise | `White Noise` | 1→0 | — | `Create(RandomPower,Hand,1,CostForTurn(0)); ExhaustSelf` | — |
| Zap | `Zap` | 1→0 | M=1 | `Channel(Lightning,M)` | — |

## Watcher (75)

| Card | Game ID | Cost | Parameters (base → first upgrade) | Base rule | Changed rule on upgrade |
|---|---|---|---|---|---|
| Alpha | `Alpha` | 1 | — | `Create(Beta,DrawPileRandom,1); ExhaustSelf` | `Innate; Create(Beta,DrawPileRandom,1); ExhaustSelf` |
| Battle Hymn | `BattleHymn` | 1 | — | `On(TurnStart,Create(Smite,Hand,1))` | `Innate; On(TurnStart,Create(Smite,Hand,1))` |
| Blasphemy | `Blasphemy` | 1 | — | `Enter(Divinity); At(NextTurnStart,Die); ExhaustSelf` | `RetainSelf; Enter(Divinity); At(NextTurnStart,Die); ExhaustSelf` |
| Bowling Bash | `BowlingBash` | 1 | D=7→10 | `Repeat(Count(LivingEnemies),Attack(Target,D))` | — |
| Brilliance | `Brilliance` | 1 | D=12→16 | `Attack(Target,WithCardBonus(D,Total(MantraGained,Combat)))` | — |
| Carve Reality | `CarveReality` | 1 | D=6→10 | `Attack(Target,D); Create(Smite,Hand,1)` | — |
| Collect | `Collect` | X | — | `At(NextTurnStarts(X),Create(Upgraded(Miracle),Hand,1)); ExhaustSelf` | `At(NextTurnStarts(Add(X,1)),Create(Upgraded(Miracle),Hand,1)); ExhaustSelf` |
| Conclude | `Conclude` | 1 | D=12→16 | `Attack(AllEnemies,D); EndTurn` | — |
| Conjure Blade | `ConjureBlade` | X | — | `Create(With(Expunger,StoredHits,X),DrawPileRandom,1); ExhaustSelf` | `Create(With(Expunger,StoredHits,Add(X,1)),DrawPileRandom,1); ExhaustSelf` |
| Consecrate | `Consecrate` | 0 | D=5→8 | `Attack(AllEnemies,D)` | — |
| Crescendo | `Crescendo` | 1→0 | — | `RetainSelf; Enter(Wrath); ExhaustSelf` | — |
| Crush Joints | `CrushJoints` | 1 | D=8→10, M=1→2 | `Attack(Target,D); If(Is(PreviousCard,Skill),Apply(Target,Vulnerable,M))` | — |
| Cut Through Fate | `CutThroughFate` | 1 | D=7→9, M=2→3 | `Attack(Target,D); Scry(M); Draw(1)` | — |
| Deceive Reality | `DeceiveReality` | 1 | B=4→7 | `Block(B); Create(Safety,Hand,1)` | — |
| Defend | `Defend_P` | 1 | B=5→8 | `Block(B)` | — |
| Deus Ex Machina | `DeusExMachina` | — | M=2→3 | `Unplayable; On(DrawSelf,Seq(Create(Miracle,Hand,M),ExhaustSelf))` | — |
| Deva Form | `DevaForm` | 3 | M=1 | `Ethereal; On(NthTurnStart(n),Energy(Mul(n,M)))` | `On(NthTurnStart(n),Energy(Mul(n,M)))` |
| Devotion | `Devotion` | 1 | M=2→3 | `On(TurnStart,Apply(Player,Mantra,M))` | — |
| Empty Body | `EmptyBody` | 1 | B=7→10 | `Block(B); Enter(Neutral)` | — |
| Empty Fist | `EmptyFist` | 1 | D=9→14 | `Attack(Target,D); Enter(Neutral)` | — |
| Empty Mind | `EmptyMind` | 1 | M=2→3 | `Draw(M); Enter(Neutral)` | — |
| Eruption | `Eruption` | 2→1 | D=9 | `Attack(Target,D); Enter(Wrath)` | — |
| Establishment | `Establishment` | 1 | M=1 | `On(RetainCard,Modify(CostOf(RetainedCard),Subtract(M),Combat))` | `Innate; On(RetainCard,Modify(CostOf(RetainedCard),Subtract(M),Combat))` |
| Evaluate | `Evaluate` | 1 | B=6→10 | `Block(B); Create(Insight,DrawPileRandom,1)` | `Block(B); Create(Upgraded(Insight),DrawPileRandom,1)` |
| Fasting | `Fasting2` | 2 | M=3→4 | `Apply(Player,Strength,M); Apply(Player,Dexterity,M); Modify(TurnStartEnergy,Subtract(1),Combat)` | — |
| Fear No Evil | `FearNoEvil` | 1 | D=8→11 | `Attack(Target,D); If(IntendsAttack(Target),Enter(Calm))` | — |
| Flurry of Blows | `FlurryOfBlows` | 0 | D=4→6 | `Attack(Target,D); InDiscard(On(StanceChanged,Move(Self,Hand)))` | — |
| Flying Sleeves | `FlyingSleeves` | 1 | D=4→6 | `RetainSelf; Repeat(2,Attack(Target,D))` | — |
| Follow-Up | `FollowUp` | 1 | D=7→11 | `Attack(Target,D); If(Is(PreviousCard,Attack),Energy(1))` | — |
| Foreign Influence | `ForeignInfluence` | 0 | — | `Let(c,Choose(Offer(AnyColorAttack,3),1),Create(c,Hand,1)); ExhaustSelf` | `Let(c,Choose(Offer(AnyColorAttack,3),1),Create(c,Hand,1,CostForTurn(0))); ExhaustSelf` |
| Foresight | `Wireheading` | 1 | M=3→4 | `On(TurnStart,Scry(M))` | — |
| Halt | `Halt` | 0 | B=3→4, M=9→14 | `Block(If(InStance(Wrath),Add(B,M),B))` | — |
| Indignation | `Indignation` | 1 | M=3→5 | `If(InStance(Wrath),Apply(AllEnemies,Vulnerable,M),Enter(Wrath))` | — |
| Inner Peace | `InnerPeace` | 1 | M=3→4 | `If(InStance(Calm),Draw(M),Enter(Calm))` | — |
| Judgment | `Judgement` | 1 | M=30→40 | `If(AtMost(HP(Target),M),SetHP(Target,0))` | — |
| Just Lucky | `JustLucky` | 0 | D=3→4, B=2→3, M=1→2 | `Scry(M); Block(B); Attack(Target,D)` | — |
| Lesson Learned | `LessonLearned` | 2 | D=10→13 | `Attack(Target,D); If(Fatal,Upgrade(Random(UpgradableDeckCards,1),Permanent)); ExhaustSelf` | — |
| Like Water | `LikeWater` | 1 | M=5→7 | `On(TurnEnd,If(InStance(Calm),Block(M)))` | — |
| Master Reality | `MasterReality` | 1→0 | — | `On(CreateCard,Upgrade(CreatedCard,Combat))` | — |
| Meditate | `Meditate` | 1 | — | `MoveAndRetain(Choose(DiscardPile,1),Hand); Enter(Calm); EndTurn` | `MoveAndRetain(Choose(DiscardPile,2),Hand); Enter(Calm); EndTurn` |
| Mental Fortress | `MentalFortress` | 1 | M=4→6 | `On(StanceChanged,Block(M))` | — |
| Nirvana | `Nirvana` | 1 | M=3→4 | `On(ScryEvent,Block(M))` | — |
| Omniscience | `Omniscience` | 4→3 | — | `Let(c,Choose(DrawPile,1),Seq(Play(c,RandomEnemy),Replay(1),Exhaust(c))); ExhaustSelf` | — |
| Perseverance | `Perseverance` | 1 | B=5→7, M=2→3 | `RetainSelf; Block(B); On(RetainedSelf,Modify(BlockOf(Self),Add(M),Combat))` | — |
| Pray | `Pray` | 1 | M=3→4 | `Apply(Player,Mantra,M); Create(Insight,DrawPileRandom,1)` | — |
| Pressure Points | `PathToVictory` | 1 | M=8→11 | `Apply(Target,Mark,M); ForEach(LivingEnemies,LoseHP(Item,Stacks(Item,Mark)))` | — |
| Prostrate | `Prostrate` | 0 | B=4, M=2→3 | `Apply(Player,Mantra,M); Block(B)` | — |
| Protect | `Protect` | 2 | B=12→16 | `RetainSelf; Block(B)` | — |
| Ragnarok | `Ragnarok` | 3 | D=5→6, M=5→6 | `Repeat(M,Attack(RandomEnemy,D))` | — |
| Reach Heaven | `ReachHeaven` | 2 | D=10→15 | `Attack(Target,D); Create(ThroughViolence,DrawPileRandom,1)` | — |
| Rushdown | `Adaptation` | 1→0 | M=2 | `On(EnterWrath,Draw(M))` | — |
| Sanctity | `Sanctity` | 1 | B=6→9, M=2 | `Block(B); If(Is(PreviousCard,Skill),Draw(M))` | — |
| Sands of Time | `SandsOfTime` | 4 | D=20→26 | `RetainSelf; Attack(Target,D); On(RetainedSelf,Modify(SelfCost,Subtract(1),Combat))` | — |
| Sash Whip | `SashWhip` | 1 | D=8→10, M=1→2 | `Attack(Target,D); If(Is(PreviousCard,Attack),Apply(Target,Weak,M))` | — |
| Scrawl | `Scrawl` | 1→0 | — | `Draw(EmptyHandSlots); ExhaustSelf` | — |
| Signature Move | `SignatureMove` | 2 | D=30→40 | `Require(Equal(Count(Filter(Hand,Attack)),1)); Attack(Target,D)` | — |
| Simmering Fury | `Vengeance` | 1 | M=2→3 | `At(NextTurnStart,Seq(Enter(Wrath),Draw(M)))` | — |
| Spirit Shield | `SpiritShield` | 2 | M=3→4 | `Block(Mul(M,Count(OtherHandCards)))` | — |
| Strike | `Strike_P` | 1 | D=6→9 | `Attack(Target,D)` | — |
| Study | `Study` | 2→1 | — | `On(TurnEnd,Create(Insight,DrawPileRandom,1))` | — |
| Swivel | `Swivel` | 2 | B=8→11 | `Block(B); NextPlayed(Attack,1,Combat,SetCost(PlayedCard,0,ThisPlay))` | — |
| Talk to the Hand | `TalkToTheHand` | 1 | D=5→7, M=2→3 | `Attack(Target,D); On(AttackTarget(Target),Block(M)); ExhaustSelf` | — |
| Tantrum | `Tantrum` | 1 | D=3, M=3→4 | `Repeat(M,Attack(Target,D)); Enter(Wrath); Move(Self,DrawPileRandom)` | — |
| Third Eye | `ThirdEye` | 1 | B=7→9, M=3→5 | `Block(B); Scry(M)` | — |
| Tranquility | `ClearTheMind` | 1→0 | — | `RetainSelf; Enter(Calm); ExhaustSelf` | — |
| Vault | `Vault` | 3→2 | — | `ExtraTurn(1); EndTurn; ExhaustSelf` | — |
| Vigilance | `Vigilance` | 2 | B=8→12 | `Block(B); Enter(Calm)` | — |
| Wallop | `Wallop` | 2 | D=9→12 | `Attack(Target,D); Block(LastUnblockedDamageTotal)` | — |
| Wave of the Hand | `WaveOfTheHand` | 1 | M=1→2 | `During(Turn,On(GainBlock,Apply(AllEnemies,Weak,M)))` | — |
| Weave | `Weave` | 0 | D=4→6 | `Attack(Target,D); InDiscard(On(ScryEvent,Move(Self,Hand)))` | — |
| Wheel Kick | `WheelKick` | 2 | D=15→20, M=2 | `Attack(Target,D); Draw(M)` | — |
| Windmill Strike | `WindmillStrike` | 2 | D=7→10, M=4→5 | `RetainSelf; Attack(Target,D); On(RetainedSelf,Modify(DamageOf(Self),Add(M),Combat))` | — |
| Wish | `Wish` | 3 | D=3→4, B=6→8, M=25→30 | `ChooseOne(Apply(Player,PlatedArmor,B),Apply(Player,Strength,D),GainGold(M)); ExhaustSelf` | — |
| Worship | `Worship` | 2 | M=5 | `Apply(Player,Mantra,M)` | `RetainSelf; Apply(Player,Mantra,M)` |
| Wreath of Flame | `WreathOfFlame` | 1 | M=5→8 | `NextPlayed(Attack,1,Combat,AddAttackDamage(M))` | — |

## Colorless (39)

| Card | Game ID | Cost | Parameters (base → first upgrade) | Base rule | Changed rule on upgrade |
|---|---|---|---|---|---|
| Apotheosis | `Apotheosis` | 2→1 | — | `Upgrade(AllCombatCards,Combat); ExhaustSelf` | — |
| Apparition | `Ghostly` | 1 | — | `Ethereal; Apply(Player,Intangible,1); ExhaustSelf` | `Apply(Player,Intangible,1); ExhaustSelf` |
| Bandage Up | `Bandage Up` | 0 | M=4→6 | `Heal(M); ExhaustSelf` | — |
| Bite | `Bite` | 1 | D=7→8, M=2→3 | `Attack(Target,D); Heal(M)` | — |
| Blind | `Blind` | 0 | M=2 | `Apply(Target,Weak,M)` | `Apply(AllEnemies,Weak,M)` |
| Chrysalis | `Chrysalis` | 2 | M=3→5 | `Create(RandomSkill,DrawPileRandom,M,CostForCombat(0)); ExhaustSelf` | — |
| Dark Shackles | `Dark Shackles` | 0 | M=9→15 | `TempApply(Target,Strength,Neg(M),Turn); ExhaustSelf` | — |
| Deep Breath | `Deep Breath` | 0 | M=1→2 | `ShuffleInto(DiscardPile,DrawPile); Draw(M)` | — |
| Discovery | `Discovery` | 1 | — | `Let(c,Choose(Offer(RandomCard,3),1),Create(c,Hand,1,CostForTurn(0))); ExhaustSelf` | `Let(c,Choose(Offer(RandomCard,3),1),Create(c,Hand,1,CostForTurn(0)))` |
| Dramatic Entrance | `Dramatic Entrance` | 0 | D=8→12 | `Innate; Attack(AllEnemies,D); ExhaustSelf` | — |
| Enlightenment | `Enlightenment` | 0 | — | `CapCost(Hand,1,Turn)` | `CapCost(Hand,1,Combat)` |
| Finesse | `Finesse` | 0 | B=2→4 | `Block(B); Draw(1)` | — |
| Flash of Steel | `Flash of Steel` | 0 | D=3→6 | `Attack(Target,D); Draw(1)` | — |
| Forethought | `Forethought` | 0 | — | `Let(c,Choose(Hand,1),Seq(Move(c,DrawPileBottom),SetCost(c,0,UntilPlayed)))` | `Let(c,ChooseAny(Hand),Seq(Move(c,DrawPileBottom),SetCost(c,0,UntilPlayed)))` |
| Good Instincts | `Good Instincts` | 0 | B=6→9 | `Block(B)` | — |
| Hand of Greed | `HandOfGreed` | 2 | D=20→25, M=20→25 | `Attack(Target,D); If(Fatal,GainGold(M))` | — |
| Impatience | `Impatience` | 0 | M=2→3 | `If(Empty(Filter(Hand,Attack)),Draw(M))` | — |
| J.A.X. | `J.A.X.` | 0 | M=2→3 | `LoseHP(Player,3); Apply(Player,Strength,M)` | — |
| Jack of All Trades | `Jack Of All Trades` | 0 | M=1→2 | `Create(RandomColorless,Hand,M); ExhaustSelf` | — |
| Madness | `Madness` | 1→0 | — | `SetCost(Random(EligibleNonzeroHandCards,1),0,Combat); ExhaustSelf` | — |
| Magnetism | `Magnetism` | 2→1 | — | `On(TurnStart,Create(RandomColorless,Hand,1))` | — |
| Master of Strategy | `Master of Strategy` | 0 | M=3→4 | `Draw(M); ExhaustSelf` | — |
| Mayhem | `Mayhem` | 2→1 | — | `On(TurnStart,Play(Top(DrawPile),RandomEnemy))` | — |
| Metamorphosis | `Metamorphosis` | 2 | M=3→5 | `Create(RandomAttack,DrawPileRandom,M,CostForCombat(0)); ExhaustSelf` | — |
| Mind Blast | `Mind Blast` | 2→1 | — | `Innate; Attack(Target,Count(DrawPile))` | — |
| Panacea | `Panacea` | 0 | M=1→2 | `Apply(Player,Artifact,M); ExhaustSelf` | — |
| Panache | `Panache` | 0 | M=10→14 | `On(EveryNthCardThisTurn(5),Damage(AllEnemies,M))` | — |
| Panic Button | `PanicButton` | 0 | B=30→40, M=2 | `Block(B); Prevent(GainBlockFromCards,Turns(M)); ExhaustSelf` | — |
| Purity | `Purity` | 0 | M=3→5 | `Exhaust(ChooseUpTo(Hand,M)); ExhaustSelf` | — |
| Ritual Dagger | `RitualDagger` | 1 | D=15, M=3→5 | `Attack(Target,D); If(Fatal,Modify(DamageOf(Self),Add(M),Permanent)); ExhaustSelf` | — |
| Sadistic Nature | `Sadistic Nature` | 0 | M=5→7 | `On(ApplyDebuff,Damage(AffectedEnemy,M))` | — |
| Secret Technique | `Secret Technique` | 0 | — | `Move(Choose(Filter(DrawPile,Skill),1),Hand); ExhaustSelf` | `Move(Choose(Filter(DrawPile,Skill),1),Hand)` |
| Secret Weapon | `Secret Weapon` | 0 | — | `Move(Choose(Filter(DrawPile,Attack),1),Hand); ExhaustSelf` | `Move(Choose(Filter(DrawPile,Attack),1),Hand)` |
| Swift Strike | `Swift Strike` | 0 | D=7→10 | `Attack(Target,D)` | — |
| The Bomb | `The Bomb` | 2 | M=40→50 | `At(NthTurnEnd(3),Damage(AllEnemies,M))` | — |
| Thinking Ahead | `Thinking Ahead` | 0 | — | `Draw(2); Move(Choose(Hand,1),DrawPileTop); ExhaustSelf` | `Draw(2); Move(Choose(Hand,1),DrawPileTop)` |
| Transmutation | `Transmutation` | X | — | `Create(RandomColorless,Hand,X,CostForTurn(0)); ExhaustSelf` | `Create(Upgraded(RandomColorless),Hand,X,CostForTurn(0)); ExhaustSelf` |
| Trip | `Trip` | 0 | M=2 | `Apply(Target,Vulnerable,M)` | `Apply(AllEnemies,Vulnerable,M)` |
| Violence | `Violence` | 0 | M=3→4 | `Move(Random(Filter(DrawPile,Attack),M),Hand); ExhaustSelf` | — |

## Curses (14)

| Card | Game ID | Cost | Parameters (base → first upgrade) | Base rule | Changed rule on upgrade |
|---|---|---|---|---|---|
| Ascender's Bane | `AscendersBane` | — | — | `Unplayable; Ethereal; CannotRemove` | — |
| Clumsy | `Clumsy` | — | — | `Unplayable; Ethereal` | — |
| Curse of the Bell | `CurseOfTheBell` | — | — | `Unplayable; CannotRemove` | — |
| Decay | `Decay` | — | — | `Unplayable; InHand(On(TurnEnd,Damage(Player,2)))` | — |
| Doubt | `Doubt` | — | — | `Unplayable; InHand(On(TurnEnd,Apply(Player,Weak,1)))` | — |
| Injury | `Injury` | — | — | `Unplayable` | — |
| Necronomicurse | `Necronomicurse` | — | — | `Unplayable; On(ExhaustSelf,Create(Copy(Self),Hand,1))` | — |
| Normality | `Normality` | — | — | `Unplayable; InHand(Cap(CardsPlayedThisTurn,3))` | — |
| Pain | `Pain` | — | — | `Unplayable; InHand(On(PlayOtherCard,LoseHP(Player,1)))` | — |
| Parasite | `Parasite` | — | — | `Unplayable; On(RemoveOrTransformSelf,LoseMaxHP(3))` | — |
| Pride | `Pride` | 1 | — | `Innate; InHand(On(TurnEnd,Create(Copy(Self),DrawPileTop,1))); ExhaustSelf` | — |
| Regret | `Regret` | — | — | `Unplayable; InHand(On(TurnEnd,LoseHP(Player,Count(Hand))))` | — |
| Shame | `Shame` | — | — | `Unplayable; InHand(On(TurnEnd,Apply(Player,Frail,1)))` | — |
| Writhe | `Writhe` | — | — | `Unplayable; Innate` | — |

## Status (5)

| Card | Game ID | Cost | Parameters (base → first upgrade) | Base rule | Changed rule on upgrade |
|---|---|---|---|---|---|
| Burn | `Burn` | — | — | `Unplayable; InHand(On(TurnEnd,Damage(Player,2)))` | `Unplayable; InHand(On(TurnEnd,Damage(Player,4)))` |
| Dazed | `Dazed` | — | — | `Unplayable; Ethereal` | — |
| Slimed | `Slimed` | 1 | — | `ExhaustSelf` | — |
| Void | `Void` | — | — | `Unplayable; Ethereal; On(DrawSelf,Energy(-1))` | — |
| Wound | `Wound` | — | — | `Unplayable` | — |

## Generated cards (9)

| Card | Game ID | Cost | Parameters (base → first upgrade) | Base rule | Changed rule on upgrade |
|---|---|---|---|---|---|
| Beta | `Beta` | 2→1 | — | `Create(Omega,DrawPileRandom,1); ExhaustSelf` | — |
| Expunger | `Expunger` | 1 | D=9→15 | `Repeat(StoredHits(Self),Attack(Target,D))` | — |
| Insight | `Insight` | 0 | M=2→3 | `RetainSelf; Draw(M); ExhaustSelf` | — |
| Miracle | `Miracle` | 0 | — | `RetainSelf; Energy(1); ExhaustSelf` | `RetainSelf; Energy(2); ExhaustSelf` |
| Omega | `Omega` | 3 | M=50→60 | `On(TurnEnd,Damage(AllEnemies,M))` | — |
| Safety | `Safety` | 1 | B=12→16 | `RetainSelf; Block(B); ExhaustSelf` | — |
| Shiv | `Shiv` | 0 | D=4→6 | `Attack(Target,D); ExhaustSelf` | — |
| Smite | `Smite` | 1 | D=12→16 | `RetainSelf; Attack(Target,D); ExhaustSelf` | — |
| Through Violence | `ThroughViolence` | 0 | D=20→30 | `RetainSelf; Attack(Target,D); ExhaustSelf` | — |

## Choice cards (3)

| Card | Game ID | Cost | Parameters (base → first upgrade) | Base rule | Changed rule on upgrade |
|---|---|---|---|---|---|
| Become Almighty | `BecomeAlmighty` | — | M=3→4 | `Apply(Player,Strength,M)` | — |
| Fame and Fortune | `FameAndFortune` | — | M=25→30 | `GainGold(M)` | — |
| Live Forever | `LiveForever` | — | M=6→8 | `Apply(Player,PlatedArmor,M)` | — |

## Other localization entries

Two auxiliary stance choices also have selection handlers in the shipped game. They sit outside CardLibrary. Choosing Calm calls `Enter(Calm)`; choosing Wrath calls `Enter(Wrath)`.

These 53 localized IDs were absent from the registered library IDs. They are kept out of the training catalogue. Some correspond to retired classes, unused designs or auxiliary stance descriptions.

`Allocate`, `Axe Kick`, `Blaster`, `Energy Pulse`, `Flux Capacitor`, `Hide`, `Mastery`, `FlowState`, `PalmThatRestrains`, `Wrath`, `Calm`, `Polymath`, `Flicker`, `FlameMastery`, `StepAndStrike`, `Survey`, `Swipe`, `Blessed`, `LetFateDecide`, `RetreatingHand`, `Flare`, `PerfectedForm`, `AwakenedStrike`, `Experienced`, `SublimeSlice`, `Prediction`, `Fasting`, `Clarity`, `Thwack`, `Flow`, `Truth`, `Wisdom`, `Peace`, `WardAura`, `Metaphysics`, `SoothingAura`, `BrillianceAura`, `FuryAura`, `CleanseEvil`, `Torrent`, `Windup`, `Shield`, `CrescentKick`, `Punishment`, `Serenity`, `Transcendence`, `Unraveling`, `BigBrain`, `Retribution`, `Causality`, `WaveOfTheHand2`, `Discipline`, `Flick`.
