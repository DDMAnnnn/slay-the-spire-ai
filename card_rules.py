"""Public card semantics. Categorical positions are part of the encoder format."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Effect:
    operation: str
    target: str = "self"
    amount: float = 0
    subject: str = "none"
    trigger: str = "play"
    source: str = "card"
    amount_from: str = "literal"
    hits: int = 1


@dataclass(frozen=True)
class Reading:
    effects: tuple[Effect, ...]
    unknown: tuple[str, ...] = ()


# Default tooltip rules describe a referenced mechanic, never extra on-play effects.
RULES = {
    "lightning": (
        Effect("damage", "random_enemy", 3, trigger="end_turn", source="orb"),
        Effect("damage", "random_enemy", 8, trigger="evoke", source="orb"),
    ),
    "vulnerable": (Effect("multiply_attack", "incoming", 1.5, trigger="while_active"),),
    "wrath": (
        Effect("multiply_attack", "incoming", 2, trigger="while_active"),
        Effect("multiply_attack", "outgoing", 2, trigger="while_active"),
    ),
}

FIELDS = {
    "operation": ("unknown", "damage", "block", "draw", "apply", "channel",
                  "enter", "multiply_attack", "gain_strength", "exhaust_self",
                  "unplayable", "ethereal", "permanent", "copy_to_discard"),
    "target": ("self", "enemy", "random_enemy", "incoming", "outgoing", "all_enemies"),
    "subject": ("none", "lightning", "vulnerable", "wrath"),
    "trigger": ("play", "on_card_exhausted", "end_turn", "evoke", "while_active",
                "in_hand", "in_deck"),
    "source": ("card", "attack", "orb", "power"),
    "amount_from": ("literal", "self_block"),
}
CARD_TYPES = ("UNKNOWN", "ATTACK", "SKILL", "POWER", "STATUS", "CURSE")

