package spireai;

import com.google.gson.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.UUID;

final class FightSession implements Closeable {
    static final Gson JSON = new Gson();
    final Path folder;
    final Path request;
    final Path reply;
    final BufferedWriter log;
    final String id = UUID.randomUUID().toString();
    int sequence;
    String ticket;
    String published;

    FightSession(Path folder, JsonObject initial) throws IOException {
        this.folder = folder;
        Files.createDirectories(folder.resolve("recordings"));
        request = folder.resolve("decision.json");
        reply = folder.resolve("action.json");
        log = Files.newBufferedWriter(folder.resolve("recordings").resolve(id + ".jsonl"),
                StandardCharsets.UTF_8, StandardOpenOption.CREATE_NEW);
        JsonObject start = new JsonObject();
        start.addProperty("schema", 2);
        start.addProperty("fight", id);
        start.add("initial", initial);
        record("start", start);
    }

    void record(String type, JsonObject row) throws IOException {
        row.addProperty("event", type);
        row.addProperty("seq", sequence++);
        log.write(JSON.toJson(row));
        log.newLine();
        log.flush();
    }

    void publish(JsonObject snapshot) throws IOException {
        published = snapshot.toString();
        ticket = id + "-" + sequence;
        snapshot.addProperty("ticket", ticket);
        record("decision", snapshot);
        Path tmp = Files.createTempFile(folder, "decision-", ".tmp");
        try {
            Files.write(tmp, JSON.toJson(snapshot).getBytes(StandardCharsets.UTF_8));
            Files.move(tmp, request, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } finally {
            Files.deleteIfExists(tmp);
        }
    }

    String readReply() throws IOException {
        if (!Files.exists(reply)) return null;
        String body = new String(Files.readAllBytes(reply), StandardCharsets.UTF_8);
        Files.deleteIfExists(reply);
        JsonObject message = new JsonParser().parse(body).getAsJsonObject();
        if (!ticket.equals(message.get("ticket").getAsString())) return null;
        return message.get("action").getAsString();
    }

    void closeWith(String outcome, JsonObject state) throws IOException {
        JsonObject end = new JsonObject();
        end.addProperty("outcome", outcome);
        end.add("state", state);
        try { record("finish", end); } finally { close(); }
    }

    public void close() throws IOException {
        try { log.close(); } finally { Files.deleteIfExists(request); }
    }
}
