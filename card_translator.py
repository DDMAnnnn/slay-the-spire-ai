"""Supervised bilingual clause classifier with copied numeric arguments."""
from dataclasses import replace
from functools import lru_cache
import re
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from card_rules import Effect, Reading, visible_description

# Supervision is at the clause level. A number in a training sentence is a
# variable copied from the query. The effect's other fields are its label.
EXAMPLES = (
    ((Effect("damage", "all_enemies", source="attack"),
      Effect("apply", "all_enemies", subject="vulnerable")),
     "Deal 4 damage and apply 1 Vulnerable to ALL enemies.", "对所有敌人造成4点伤害，给予1层易伤。"),
    (Effect("damage", "enemy", source="attack"), "Deal 6 damage.", "造成6点伤害。"),
    (Effect("damage", "enemy", source="attack", hits=2), "Deal 5 damage twice.", "造成5点伤害两次。"),
    (Effect("damage", "enemy", source="attack", amount_from="self_block"),
     "Deal damage equal to your Block.", "造成你当前格挡值的伤害。"),
    (Effect("block"), "Gain 5 Block.", "获得5点格挡。"),
    (Effect("draw"), "Draw 1 card.", "Draw 2 cards.", "抽1张牌。"),
    (Effect("apply", "enemy", subject="vulnerable"), "Apply 2 Vulnerable.", "给予2层易伤。"),
    (Effect("gain_strength"), "Gain 2 Strength.", "获得2点力量。"),
    (Effect("block", trigger="on_card_exhausted", source="power"),
     "Whenever a card is Exhausted, gain 3 Block.", "每当有一张牌被消耗时，获得3点格挡。"),
    (Effect("channel", subject="lightning"), "Channel 1 Lightning.", "生成1个闪电充能球。"),
    (Effect("enter", subject="wrath"), "Enter Wrath.", "进入愤怒。"),
    (Effect("copy_to_discard", amount=1), "Add a copy of this card into your discard pile.",
     "在弃牌堆放入一张此牌的复制品。"),
    (Effect("unplayable", trigger="in_hand"), "Unplayable.", "不能被打出。"),
    (Effect("ethereal", trigger="end_turn"), "Ethereal.", "虚无。"),
    (Effect("permanent", trigger="in_deck"), "Cannot be removed from your deck.", "不能从牌组中移除。"),
    (Effect("exhaust_self"), "Exhaust.", "消耗。"),
)

def tokens(text):
    return tuple(re.findall(r"[a-z]+|[0-9]+|[\u3400-\u9fff]|[^\s]", text.lower()))


def parts(text):
    text = unicodedata.normalize("NFKC", text)
    # NL is a display wrap, including in the middle of conditional clauses.
    text = re.sub(r"\bNL\b", " ", text)
    return [p.strip() for p in re.split(r"[.。]", text) if p.strip()]


def signature(words):
    return tuple("#" if word.isdecimal() else word for word in words)


class Translator:
    def __init__(self, examples=EXAMPLES):
        forms = {}
        self.classes = []
        self.transitions = []
        self.arities = []
        for index, (label, *samples) in enumerate(examples):
            program = label if isinstance(label, tuple) else (label,)
            self.classes.append(program)
            transitions = set()
            arities = set()
            for sample in samples:
                clause, = parts(sample)
                key = signature(tokens(clause))
                if key.count("#") not in (0, len(program)):
                    raise ValueError(f"Ambiguous number binding: {sample}")
                if forms.setdefault(key, index) != index:
                    raise ValueError(f"Conflicting supervision: {sample}")
                transitions.update(zip(("<start>",) + key, key + ("<end>",)))
                arities.add(key.count("#"))
            self.transitions.append(transitions)
            self.arities.append(arities)
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 4))
        samples = self.vectorizer.fit_transform([" ".join(form) for form in forms])
        self.model = LogisticRegression(C=100, max_iter=500).fit(samples, list(forms.values()))

    def classify(self, key):
        # A new qualifier, reordered condition or extra numeric argument is
        # evidence outside the supervised grammar. Retain it for replay.
        features = self.vectorizer.transform([" ".join(key)])
        probabilities = self.model.predict_proba(features)[0]
        best = probabilities.argmax()
        if probabilities[best] < 0.80:
            return None
        label = self.model.classes_[best]
        transitions = zip(("<start>",) + key, key + ("<end>",))
        if key.count("#") not in self.arities[label] or any(
                pair not in self.transitions[label] for pair in transitions):
            return None
        return self.classes[label]

    def read(self, card):
        return self.translate(visible_description(card))

    @lru_cache(maxsize=2048)
    def translate(self, text):
        effects, missing = [], []
        if not text.strip():
            return Reading((Effect("unknown"),), ("missing description",))
        # Body Slam's parenthetical is a display of the preceding rule's
        # current value, not a second hit. Only fold this exact annotation.
        text = re.sub(r"(Deal damage equal to your Block\.)\s*(?:NL\s*)?\(Deals? \d+ damage\.\)", r"\1", text, flags=re.I)
        text = re.sub(r"(造成你当前\s*格挡\s*值的伤害。)\s*(?:NL\s*)?[（(]造成\s*\d+\s*点伤害。[）)]", r"\1", text)
        for clause in parts(text):
            words = tokens(clause)
            label = self.classify(signature(words))
            if label is None:
                effects.append(Effect("unknown"))
                missing.append(clause)
                continue
            numbers = [int(w) for w in words if w.isdecimal()]
            if numbers:
                effects.extend(replace(effect, amount=n) for effect, n in zip(label, numbers, strict=True))
            else:
                effects.extend(label)
        if not effects:
            return Reading((Effect("unknown"),), (text,))
        return Reading(tuple(effects), tuple(missing))
