package spireai;

import com.google.gson.*;
import com.megacrit.cardcrawl.actions.GameActionManager;
import com.megacrit.cardcrawl.cards.*;
import com.megacrit.cardcrawl.characters.AbstractPlayer;
import com.megacrit.cardcrawl.core.AbstractCreature;
import com.megacrit.cardcrawl.dungeons.AbstractDungeon;
import com.megacrit.cardcrawl.helpers.GameDictionary;
import com.megacrit.cardcrawl.monsters.AbstractMonster;
import com.megacrit.cardcrawl.potions.*;
import com.megacrit.cardcrawl.orbs.AbstractOrb;
import com.megacrit.cardcrawl.powers.AbstractPower;
import com.megacrit.cardcrawl.relics.AbstractRelic;
import com.megacrit.cardcrawl.ui.panels.EnergyPanel;
import basemod.ReflectionHacks;
import java.util.*;

final class CombatFrame {
    static final Gson JSON = new Gson();
    final JsonObject state = new JsonObject();
    final Map<String,Runnable> moves = new LinkedHashMap<>();

    static JsonObject creature(AbstractCreature c) {
        JsonObject obj = new JsonObject();
        obj.addProperty("name", c.name);
        obj.addProperty("current_hp", c.currentHealth);
        obj.addProperty("max_hp", c.maxHealth);
        obj.addProperty("block", c.currentBlock);
        JsonArray powers = new JsonArray();
        for (AbstractPower p : c.powers) {
            JsonObject power = new JsonObject();
            power.addProperty("id", p.ID);
            power.addProperty("amount", p.amount);
            power.addProperty("description", p.description);
            powers.add(power);
        }
        obj.add("powers", powers);
        return obj;
    }

    static JsonObject card(AbstractCard c) {
        JsonObject obj = new JsonObject();
        obj.addProperty("id", c.cardID);
        obj.addProperty("name", c.name);
        obj.addProperty("upgrades", c.timesUpgraded);
        obj.addProperty("description", c.rawDescription);
        JsonObject glossary = new JsonObject();
        for (String keyword : c.keywords) {
            String definition = GameDictionary.keywords.get(keyword);
            if (definition != null) glossary.addProperty(keyword, definition);
        }
        obj.add("glossary", glossary);
        obj.addProperty("type", c.type.name());
        obj.addProperty("target", c.target.name());
        obj.addProperty("base_damage", c.baseDamage);
        obj.addProperty("base_block", c.baseBlock);
        obj.addProperty("base_magic", c.baseMagicNumber);
        obj.addProperty("cost", c.costForTurn);
        obj.addProperty("damage", c.damage);
        obj.addProperty("block", c.block);
        obj.addProperty("magic", c.magicNumber);
        return obj;
    }

    static JsonArray pile(CardGroup group) {
        List<JsonObject> list = new ArrayList<>();
        for (AbstractCard c : group.group) list.add(card(c));
        list.sort(Comparator.comparing(JsonObject::toString));
        return JSON.toJsonTree(list).getAsJsonArray();
    }

    static JsonObject inventory(AbstractPlayer p) {
        JsonObject inv = creature(p);
        inv.addProperty("class", p.chosenClass.name());
        inv.addProperty("ascension", AbstractDungeon.ascensionLevel);
        inv.addProperty("floor", AbstractDungeon.floorNum);
        inv.addProperty("gold", p.gold);
        inv.addProperty("stance", p.stance.ID);
        JsonArray orbs = new JsonArray();
        for (AbstractOrb orb : p.orbs) {
            JsonObject slot = new JsonObject();
            slot.addProperty("id", orb.ID);
            slot.addProperty("passive", orb.passiveAmount);
            slot.addProperty("evoke", orb.evokeAmount);
            orbs.add(slot);
        }
        inv.add("orb_slots", orbs);
        inv.add("deck", pile(p.masterDeck));
        JsonArray potions = new JsonArray();
        JsonArray potionDetails = new JsonArray();
        for (AbstractPotion potion : p.potions) {
            potions.add(potion.ID);
            JsonObject detail = new JsonObject();
            detail.addProperty("id", potion.ID);
            detail.addProperty("description", potion.description);
            if (!(potion instanceof PotionSlot))
                detail.addProperty("potency", ReflectionHacks.<Integer>getPrivate(potion, AbstractPotion.class, "potency"));
            potionDetails.add(detail);
        }
        inv.add("potions", potions);
        inv.add("potion_details", potionDetails);
        JsonArray relics = new JsonArray();
        for (AbstractRelic r : p.relics) {
            JsonObject relic = new JsonObject();
            relic.addProperty("id", r.relicId);
            relic.addProperty("counter", r.counter);
            relic.addProperty("description", r.description);
            relics.add(relic);
        }
        inv.add("relics", relics);
        return inv;
    }

    CombatFrame(AbstractPlayer p) {
        state.add("player", inventory(p));
        state.addProperty("turn", GameActionManager.turn);
        state.addProperty("energy", EnergyPanel.totalCount);
        JsonArray hand = new JsonArray();
        for (AbstractCard c : p.hand.group) hand.add(card(c));
        state.add("hand", hand);
        state.add("draw", pile(p.drawPile));
        state.add("discard", pile(p.discardPile));
        state.add("exhaust", pile(p.exhaustPile));
        List<AbstractMonster> enemies = AbstractDungeon.getMonsters().monsters;
        JsonArray monsters = new JsonArray();
        for (AbstractMonster m : enemies) {
            JsonObject obj = creature(m);
            obj.addProperty("id", m.id);
            obj.addProperty("targetable", !m.isDeadOrEscaped() && !m.halfDead);
            if (!p.hasRelic("Runic Dome")) {
                obj.addProperty("intent", m.intent.name());
                if (m.intent.name().startsWith("ATTACK")) {
                    obj.addProperty("damage", m.getIntentDmg());
                    boolean multi = ReflectionHacks.getPrivate(m, AbstractMonster.class, "isMultiDmg");
                    int hits = multi ? ReflectionHacks.<Integer>getPrivate(m, AbstractMonster.class, "intentMultiAmt") : 1;
                    obj.addProperty("hits", hits);
                }
            }
            monsters.add(obj);
        }
        state.add("monsters", monsters);

        for (int i = 0; i < p.hand.size(); i++) {
            AbstractCard c = p.hand.group.get(i);
            String key = "play " + i;
            if (c.target == AbstractCard.CardTarget.ENEMY || c.target == AbstractCard.CardTarget.SELF_AND_ENEMY) {
                for (int j = 0; j < enemies.size(); j++) {
                    AbstractMonster m = enemies.get(j);
                    if (!m.isDeadOrEscaped() && !m.halfDead && c.canUse(p, m))
                        moves.put(key + " " + j, () -> play(c, m));
                }
            } else if (c.canUse(p, null)) moves.put(key, () -> play(c, null));
        }
        for (AbstractPotion potion : p.potions) {
            if (potion instanceof PotionSlot || !potion.canUse()) continue;
            String key = "potion " + potion.slot;
            if (potion.targetRequired) {
                for (int j = 0; j < enemies.size(); j++) {
                    AbstractMonster m = enemies.get(j);
                    if (!m.isDeadOrEscaped() && !m.halfDead)
                        moves.put(key + " " + j, () -> drink(p, potion, m));
                }
            } else moves.put(key, () -> drink(p, potion, p));
        }
        moves.put("end", () -> AbstractDungeon.overlayMenu.endTurnButton.disable(true));
    }

    private static void play(AbstractCard card, AbstractMonster target) {
        AbstractDungeon.actionManager.addCardQueueItem(new CardQueueItem(card, target, EnergyPanel.totalCount));
    }

    private static void drink(AbstractPlayer p, AbstractPotion potion, AbstractCreature target) {
        potion.use(target);
        for (AbstractRelic relic : p.relics) relic.onUsePotion();
        AbstractDungeon.topPanel.destroyPotion(potion.slot);
    }
}
