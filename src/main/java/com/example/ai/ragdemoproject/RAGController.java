package com.example.ai.ragdemoproject;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.web.bind.annotation.*;
import java.util.*;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/rag")
public class RAGController {

    private final ChatClient chatClient;
    private final EmbeddingModel embeddingModel;
    private final List<Doc> db = new ArrayList<>();

    record Doc(String text, float[] embedding) {}

    public RAGController(ChatClient.Builder builder, EmbeddingModel embeddingModel) {
        this.chatClient = builder.build();
        this.embeddingModel = embeddingModel;
    }

    @PostMapping("/add")
    public String add(@RequestBody Map<String, String> body) {
        String text = body.get("text");
        float[] emb = embeddingModel.embed(text);
        db.add(new Doc(text, emb));
        return "Added: " + text + " | DB size: " + db.size();
    }

    @GetMapping("/ask")
    public String ask(@RequestParam String q) {
        float[] qEmb = embeddingModel.embed(q);

        // cosine similarity search - TOP 3
        List<Doc> top = db.stream()
                .sorted((a,b) -> Float.compare(cosine(b.embedding, qEmb), cosine(a.embedding, qEmb)))
                .limit(3)
                .toList();

        String context = top.stream().map(d -> d.text).collect(Collectors.joining("\n"));
        System.out.println("CONTEXT FOUND: " + context);

        if (context.isBlank()) return "No data in vector DB yet. POST /api/rag/add first.";

        return chatClient.prompt()
                .system("You are a helpful assistant. Answer ONLY from CONTEXT:\n" + context)
                .user(q)
                .call()
                .content();
    }

    float cosine(float[] a, float[] b) {
        float dot=0, na=0, nb=0;
        for (int i=0;i<a.length;i++) {
            dot+=a[i]*b[i];
            na+=a[i]*a[i];
            nb+=b[i]*b[i];
        }
        return dot / (float)(Math.sqrt(na)*Math.sqrt(nb) + 1e-9);
    }
}