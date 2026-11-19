package spireai;

import com.google.gson.*;
import java.nio.file.*;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

/** Exercise the production mailbox and recorder without touching a save file. */
public class ProtocolCheck {
    public static void main(String[] args) throws Exception {
        Path dir = Paths.get(args[0]);
        JsonObject state = new JsonParser().parse("{\"turn\":1,\"energy\":3,\"player\":{\"current_hp\":70,\"block\":0,\"potions\":[]},\"hand\":[],\"monsters\":[]}").getAsJsonObject();
        FightSession session = new FightSession(dir, state);
        JsonObject request = new JsonObject();
        request.add("state", state);
        request.add("options", FightSession.JSON.toJsonTree(Arrays.asList("end")));
        session.publish(request);
        Files.write(session.reply, "{\"ticket\":\"old\",\"action\":\"end\"}".getBytes(StandardCharsets.UTF_8));
        if (session.readReply() != null) throw new AssertionError("Accepted stale command");
        Process policy = new ProcessBuilder(args[1], "-c",
                "from pathlib import Path; from agent import respond; import sys; respond(Path(sys.argv[1]))",
                dir.toString()).inheritIO().start();
        if (policy.waitFor() != 0 || !"end".equals(session.readReply()))
            throw new AssertionError("Round trip failed");
        JsonObject action = new JsonObject();
        action.addProperty("ticket", session.ticket);
        action.addProperty("action", "end");
        session.record("action_requested", action);
        session.closeWith("offline protocol check", state);
        if (Files.exists(session.request)) throw new AssertionError("Stale decision left behind");
        System.out.println(dir.resolve("recordings").resolve(session.id + ".jsonl"));
    }
}
