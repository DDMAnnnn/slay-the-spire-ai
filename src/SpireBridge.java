package spireai;

import basemod.BaseMod;
import basemod.interfaces.PostDungeonUpdateSubscriber;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Input;
import com.evacipated.cardcrawl.modthespire.lib.SpireInitializer;
import com.google.gson.*;
import com.megacrit.cardcrawl.actions.GameActionManager;
import com.megacrit.cardcrawl.characters.AbstractPlayer;
import com.megacrit.cardcrawl.dungeons.AbstractDungeon;
import com.megacrit.cardcrawl.rooms.AbstractRoom;
import java.io.IOException;
import java.nio.file.*;

@SpireInitializer
public class SpireBridge implements PostDungeonUpdateSubscriber {
    private FightSession fight;
    private AbstractRoom originalRoom;
    private long nextPoll;
    private static final Path HOME = Paths.get(System.getProperty("user.home"), "SpireCombatAI");

    public static void initialize() { BaseMod.subscribe(new SpireBridge()); }

    public void receivePostDungeonUpdate() {
        try { update(); }
        catch (Exception e) {
            System.err.println("[SpireCombatAI] " + e);
            try { stop("interrupted: " + e.getClass().getSimpleName()); }
            catch (IOException failure) { System.err.println(failure); }
        }
    }

    private void stop(String outcome) throws IOException {
        if (fight == null) return;
        FightSession ended = fight;
        fight = null;
        AbstractPlayer p = AbstractDungeon.player;
        ended.closeWith(outcome, p == null ? null : CombatFrame.inventory(p));
        System.out.println("[SpireCombatAI] " + outcome + " — " + ended.id);
    }

    private void update() throws IOException {
        boolean toggle = Gdx.input.isKeyJustPressed(Input.Keys.F8);
        if (fight == null && !toggle) return;
        AbstractPlayer p = AbstractDungeon.player;
        AbstractRoom room = AbstractDungeon.currMapNode == null ? null : AbstractDungeon.getCurrRoom();
        if (fight != null) {
            if (toggle) { stop("stopped by player"); return; }
            if (room != originalRoom || p == null) { stop("room changed"); return; }
            if (p.isDead) { stop("defeat"); return; }
            if (room.isBattleOver || room.phase == AbstractRoom.RoomPhase.COMPLETE) {
                stop("combat ended"); return;
            }
        } else {
            if (!toggle || p == null || room == null || room.phase != AbstractRoom.RoomPhase.COMBAT
                    || room.isBattleOver || AbstractDungeon.isScreenUp) return;
            originalRoom = room;
            fight = new FightSession(HOME, CombatFrame.inventory(p));
            System.out.println("[SpireCombatAI] recording " + fight.id);
        }

        long now = System.nanoTime();
        if (now < nextPoll) return;
        nextPoll = now + 150_000_000L;
        GameActionManager manager = AbstractDungeon.actionManager;
        if (AbstractDungeon.isScreenUp) {
            // The basic agent cannot resolve modal selections. Give control back.
            stop("selection screen: " + AbstractDungeon.screen);
            return;
        }
        if (room.phase != AbstractRoom.RoomPhase.COMBAT || manager == null ||
                manager.phase != GameActionManager.Phase.WAITING_ON_USER || manager.turnHasEnded ||
                manager.currentAction != null || !manager.actions.isEmpty() || !manager.cardQueue.isEmpty() ||
                p.isEndingTurn || p.endTurnQueued || p.isDraggingCard) return;

        CombatFrame frame = new CombatFrame(p);
        JsonObject snapshot = new JsonObject();
        snapshot.add("state", frame.state);
        snapshot.add("options", FightSession.JSON.toJsonTree(frame.moves.keySet()));
        if (!snapshot.toString().equals(fight.published)) {
            fight.publish(snapshot);
            return;
        }
        String action = fight.readReply();
        if (action == null) return;
        Runnable execute = frame.moves.get(action);
        if (execute == null) { stop("illegal agent reply: " + action); return; }
        JsonObject chosen = new JsonObject();
        chosen.addProperty("ticket", fight.ticket);
        chosen.addProperty("action", action);
        fight.record("action_requested", chosen);
        fight.published = null;
        execute.run();
    }
}
