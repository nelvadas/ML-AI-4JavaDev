package com.example;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.FileSystemDocumentLoader;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.chroma.ChromaEmbeddingStore;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import static dev.langchain4j.internal.Utils.randomUUID;
import static java.time.Duration.ofSeconds;


/**
 * Hello world with langchain4j!
 *
 */
public class RAGDemoApp
{
    
    public static void main(String[] args )
    { 





        /* Online
        EmbeddingModel embeddingModel = HuggingFaceEmbeddingModel.builder()
                .accessToken(ApiKeys.HF_API_KEY)
                .modelId("sentence-transformers/all-MiniLM-L6-v2")
                .waitForModel(true)
                .timeout(ofSeconds(60))
                .build(); */

        EmbeddingModel model = new AllMiniLmL6V2EmbeddingModel();

        // make sure you have a chroma db listening on port 8000
        // podman  run -d -p 8000:8000 ghcr.io/chroma-core/chroma:0.4.6

        EmbeddingStore<TextSegment> embeddingStore = ChromaEmbeddingStore.builder()
                .baseUrl("http://localhost:8000")
                .collectionName(randomUUID())
                .build();

        Path filePath = Paths.get("src/main/resources/input.txt");
        Document document = FileSystemDocumentLoader.loadDocument(filePath);
        String[] lines = document.text().split("\\n");
        for (String line : lines) {
            TextSegment segment = TextSegment.from(line);
            Embedding lineEmbedding = model.embed(segment).content();
            String uuid = embeddingStore.add(lineEmbedding);
            System.out.println(uuid+"=> "+line);
        }

        Embedding queryEmbedding = model.embed("What is my name?").content();
        System.out.println("Query Vector" +queryEmbedding);

        List<EmbeddingMatch<TextSegment>> relevant = embeddingStore.findRelevant(queryEmbedding, 1);
        EmbeddingMatch<TextSegment> embeddingMatch = relevant.get(0);

        System.out.println("Match Score" + embeddingMatch.score());
        System.out.println("Vector similar with the query /n"+ embeddingMatch.embeddingId());




    }


    
}
