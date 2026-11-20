package spireai;

import com.google.gson.JsonObject;
import com.megacrit.cardcrawl.characters.AbstractPlayer;
import com.megacrit.cardcrawl.core.AbstractCreature;
import com.megacrit.cardcrawl.monsters.AbstractMonster;
import java.io.IOException;
import java.util.IdentityHashMap;
import java.util.List;

/** Changes in visible HP between game updates while a requested action resolves. */
final class CombatHealthTrace {
    private final IdentityHashMap<AbstractCreature, Integer> previous = new IdentityHashMap<>();
    private String ticket;

    void begin(String actionTicket, AbstractPlayer player, List<AbstractMonster> enemies) {
        previous.clear();
        previous.put(player, player.currentHealth);
        for (AbstractMonster m : enemies) previous.put(m, m.currentHealth);
        ticket = actionTicket;
    }

    void stop() {
        previous.clear();
        ticket = null;
    }

    void observe(FightSession fight, AbstractPlayer player, List<AbstractMonster> enemies) throws IOException {
        if (ticket == null) return;
        record(fight, player, -1);
        for (int i = 0; i < enemies.size(); i++) record(fight, enemies.get(i), i);
    }

    private void record(FightSession fight, AbstractCreature target, int slot) throws IOException {
        Integer before = previous.put(target, target.currentHealth);
        if (before == null || before == target.currentHealth) return;
        JsonObject change = new JsonObject();
        change.addProperty("ticket", ticket);
        change.addProperty("target", slot < 0 ? "player" : "enemy");
        if (slot >= 0) change.addProperty("slot", slot);
        change.addProperty("before_hp", before);
        change.addProperty("after_hp", target.currentHealth);
        fight.record("hp_observed", change);
    }
}
