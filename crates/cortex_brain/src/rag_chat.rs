use pgvector::Vector;
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Row};
use std::error::Error;
use std::io::{self, Write};

#[derive(Serialize)]
struct EmbedReq {
    model: String,
    prompt: String,
}

#[derive(Deserialize)]
struct EmbedRes {
    embedding: Vec<f32>,
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatReq {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
}

#[derive(Deserialize)]
struct ChatResMessage {
    content: String,
}

#[derive(Deserialize)]
struct ChatRes {
    message: ChatResMessage,
}

pub struct RagChat {
    pool: PgPool,
    client: reqwest::Client,
}

impl RagChat {
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool,
            client: reqwest::Client::new(),
        }
    }

    /// Extrage vectorul întrebării
    async fn get_query_embedding(&self, query: &str) -> Result<Vec<f32>, Box<dyn Error>> {
        let req = EmbedReq {
            model: "nomic-embed-text".to_string(),
            prompt: query.to_string(),
        };

        let res = self
            .client
            .post("http://localhost:11434/api/embeddings")
            .json(&req)
            .send()
            .await?
            .json::<EmbedRes>()
            .await?;

        Ok(res.embedding)
    }

    /// Caută fragmentele cele mai relevante în brain.memories
    /// Caută fragmentele cele mai relevante în brain.memories
    async fn retrieve_context(&self, project_id: i64, query_vector: Vec<f32>) -> Result<Vec<String>, Box<dyn Error>> {
        let vector = Vector::from(query_vector);

        let rows = sqlx::query(
            r#"
            SELECT content, content_type, (1 - (embedding <=> $1)) AS similarity
            FROM brain.memories
            WHERE project_id = $2
            ORDER BY embedding <=> $1
            LIMIT 8
            "#,
        )
        .bind(vector)
        .bind(project_id)
        .fetch_all(&self.pool)
        .await?;

        let mut contexts = Vec::new();
        for row in rows {
            let content: String = row.get("content");
            let content_type: String = row.get("content_type");
            let similarity: f64 = row.get("similarity");

            // Filtrăm rezultatele cu potrivire slabă
            if similarity >= 0.60 {
                println!("   🔍 Found match [{:.2}%]: {}...", similarity * 100.0, content.lines().next().unwrap_or(""));
                contexts.push(format!("[{}] {}", content_type, content));
            }
        }

        Ok(contexts)
    }

    /// Generează răspunsul final prin Ollama Chat
    /// Generează răspunsul final prin Ollama Chat
    async fn generate_answer(&self, prompt: &str) -> Result<String, Box<dyn Error>> {
        let system_instruction = 
            "Ești un asistent expert în C++. Analizează atent contextul primit din baza de date. "
            .to_string() + 
            "Răspunde strict pe baza funcțiilor și claselor menționate în context. " +
            "Dacă contextul conține doar nume de clase fără metode explicite, menționează clasele relevante fără să inventezi metode C++ inexistente.";

        let req = ChatReq {
            model: "llama3.2".to_string(), // sau "qwen2.5"
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system_instruction,
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: prompt.to_string(),
                },
            ],
            stream: false,
        };

        let res = self
            .client
            .post("http://localhost:11434/api/chat")
            .json(&req)
            .send()
            .await?
            .json::<ChatRes>()
            .await?;

        Ok(res.message.content)
    }

    /// Pornește bucla interactivă REPL
    pub async fn start_repl(&self, project_id: i64) -> Result<(), Box<dyn Error>> {
        println!("\n========================================================");
        println!("  🧠 Cortex Brain AI REPL (Tastați 'exit' pentru ieșire)");
        println!("========================================================\n");

        loop {
            print!("Cortex > ");
            io::stdout().flush()?;

            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let query = input.trim();

            if query.eq_ignore_ascii_case("exit") || query.is_empty() {
                println!("La revedere!");
                break;
            }

            println!("\n[1/2] Caut memorie semantică în Postgres...");
            let query_vector = match self.get_query_embedding(query).await {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("Eroare la generarea embedding-ului: {}", e);
                    continue;
                }
            };

            let context_list = match self.retrieve_context(project_id, query_vector).await {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Eroare la extragerea contextului: {}", e);
                    continue;
                }
            };

            let context_str = context_list.join("\n---\n");
            let full_prompt = format!(
                "CONTEXT DIN PROIECT:\n{}\n\nÎNTREBARE UTILIZATOR: {}\n\nFurnizează un răspuns clar și direct bazat pe contextul de mai sus.",
                context_str, query
            );

            println!("\n[2/2] Generez răspuns cu LLM...");
            match self.generate_answer(&full_prompt).await {
                Ok(answer) => println!("\n🤖 RĂSPUNS CORTEX:\n{}\n\n", answer),
                Err(e) => eprintln!("Eroare la generarea răspunsului: {}", e),
            }
        }

        Ok(())
    }
}