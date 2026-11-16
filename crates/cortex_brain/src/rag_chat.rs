use crate::config::{ollama_chat_model, ollama_url};
use crate::indexer::CodeIndexer;
use pgvector::Vector;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Row};
use std::error::Error;
use std::io;
use std::path::{Path, PathBuf};

#[derive(Serialize)]
struct EmbedReq {
    model: String,
    prompt: String,
}

#[derive(Deserialize)]
struct EmbedRes {
    embedding: Vec<f32>,
}

#[derive(Clone, Serialize)]
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

const MAX_HISTORY_MESSAGES: usize = 20;

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
            .post(ollama_url("/api/embeddings"))
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
    async fn generate_answer(
        &self,
        prompt: &str,
        history: &[ChatMessage],
    ) -> Result<String, Box<dyn Error>> {
        let system_instruction = "Ești un asistent de programare. Folosește istoricul conversației și contextul indexat al proiectului pentru a înțelege întrebările de continuare. Bazează răspunsurile pe codul și documentația disponibile; dacă lipsesc detalii, explică ipotezele în loc să pretinzi că ele există în proiect.";

        let req = ChatReq {
            model: ollama_chat_model(),
            messages: std::iter::once(ChatMessage {
                role: "system".to_string(),
                content: system_instruction.to_string(),
            })
            .chain(history.iter().cloned())
            .chain(std::iter::once(ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }))
            .collect(),
            stream: false,
        };

        let res = self
            .client
            .post(ollama_url("/api/chat"))
            .json(&req)
            .send()
            .await?
            .json::<ChatRes>()
            .await?;

        Ok(res.message.content)
    }

    async fn ensure_chat_history_table(&self) -> Result<(), Box<dyn Error>> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS brain.chat_history (
                id BIGSERIAL PRIMARY KEY,
                project_id BIGINT NOT NULL REFERENCES brain.projects(id) ON DELETE CASCADE,
                role VARCHAR(16) NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            "#,
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn load_chat_history(
        &self,
        project_id: i64,
    ) -> Result<Vec<ChatMessage>, Box<dyn Error>> {
        let rows = sqlx::query(
            r#"
            SELECT role, content
            FROM brain.chat_history
            WHERE project_id = $1
            ORDER BY id DESC
            LIMIT $2
            "#,
        )
        .bind(project_id)
        .bind(MAX_HISTORY_MESSAGES as i64)
        .fetch_all(&self.pool)
        .await?;

        let mut history: Vec<ChatMessage> = rows
            .into_iter()
            .map(|row| ChatMessage {
                role: row.get("role"),
                content: row.get("content"),
            })
            .collect();
        history.reverse();
        Ok(history)
    }

    async fn save_chat_exchange(
        &self,
        project_id: i64,
        user_prompt: &str,
        assistant_answer: &str,
    ) -> Result<(), Box<dyn Error>> {
        let mut tx = self.pool.begin().await?;
        sqlx::query(
            r#"
            INSERT INTO brain.chat_history (project_id, role, content)
            VALUES ($1, 'user', $2), ($1, 'assistant', $3)
            "#,
        )
        .bind(project_id)
        .bind(user_prompt)
        .bind(assistant_answer)
        .execute(&mut *tx)
        .await?;

        sqlx::query(
            r#"
            DELETE FROM brain.chat_history
            WHERE project_id = $1
              AND id NOT IN (
                  SELECT id
                  FROM brain.chat_history
                  WHERE project_id = $1
                  ORDER BY id DESC
                  LIMIT $2
              )
            "#,
        )
        .bind(project_id)
        .bind(MAX_HISTORY_MESSAGES as i64)
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(())
    }

    async fn handle_command(
        &self,
        input: &str,
        active_project_id: &mut i64,
    ) -> Result<bool, Box<dyn Error>> {
        let words: Vec<&str> = input.split_whitespace().collect();
        let Some(command) = words.first().map(|word| word.to_ascii_lowercase()) else {
            return Ok(false);
        };

        match command.as_str() {
            "help" => {
                println!("Comenzi disponibile:");
                println!("  reindex project <nume> - reindexează proiectul");
                println!("  delete project <nume>  - șterge proiectul și datele indexate");
                println!("  add project <nume> <cale> - adaugă și indexează un proiect");
                println!("  use project <nume>    - selectează proiectul pentru chat");
                println!("  projects               - listează proiectele");
                println!("  help                   - afișează această listă");
                println!("  exit                   - închide chatul");
                Ok(true)
            }
            "projects" if words.len() == 1 => {
                let rows = sqlx::query("SELECT name FROM brain.projects ORDER BY name")
                    .fetch_all(&self.pool)
                    .await?;
                if rows.is_empty() {
                    println!("Nu există proiecte.");
                } else {
                    println!("Proiecte:");
                    for row in rows {
                        let name: String = row.get("name");
                        println!("  - {}", name);
                    }
                }
                Ok(true)
            }
            "add" => {
                if words.len() < 4 || !words[1].eq_ignore_ascii_case("project") {
                    println!("Utilizare: add project <nume> <cale>");
                    return Ok(true);
                }

                let project_name = words[2];
                let source_path = words[3..]
                    .join(" ")
                    .trim_matches('"')
                    .trim_matches('\'')
                    .to_string();

                if !Path::new(&source_path).is_dir() {
                    println!("Directorul proiectului nu există: {}", source_path);
                    return Ok(true);
                }

                let existing_id = sqlx::query_scalar::<_, i64>(
                    "SELECT id FROM brain.projects WHERE LOWER(name) = LOWER($1)",
                )
                .bind(project_name)
                .fetch_optional(&self.pool)
                .await?;
                if existing_id.is_some() {
                    println!(
                        "Proiectul '{}' există deja. Folosește 'reindex project {}'.",
                        project_name, project_name
                    );
                    return Ok(true);
                }

                let project_id = sqlx::query_scalar::<_, i64>(
                    "INSERT INTO brain.projects (name, root_path) VALUES ($1, $2) RETURNING id",
                )
                .bind(project_name)
                .bind(&source_path)
                .fetch_one(&self.pool)
                .await?;

                if let Err(error) = CodeIndexer::new(self.pool.clone())
                    .reindex_and_store(project_id, &source_path)
                    .await
                {
                    sqlx::query("DELETE FROM brain.projects WHERE id = $1")
                        .bind(project_id)
                        .execute(&self.pool)
                        .await?;
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!("Indexarea proiectului '{}' a eșuat: {}", project_name, error),
                    )
                    .into());
                }

                *active_project_id = project_id;
                println!(
                    "Proiectul '{}' a fost adăugat, indexat și selectat pentru chat.",
                    project_name
                );
                Ok(true)
            }
            "use" => {
                if words.len() < 3 || !words[1].eq_ignore_ascii_case("project") {
                    println!("Utilizare: use project <nume>");
                    return Ok(true);
                }

                let project_name = words[2..].join(" ");
                let project = sqlx::query(
                    "SELECT id, name FROM brain.projects WHERE LOWER(name) = LOWER($1)",
                )
                .bind(&project_name)
                .fetch_optional(&self.pool)
                .await?;

                if let Some(project) = project {
                    *active_project_id = project.get("id");
                    let name: String = project.get("name");
                    println!("Proiectul '{}' este acum selectat pentru chat.", name);
                } else {
                    println!("Proiectul '{}' nu există.", project_name);
                }
                Ok(true)
            }
            "reindex" | "delete" => {
                if words.len() < 3 || !words[1].eq_ignore_ascii_case("project") {
                    println!("Utilizare: {} project <nume>", command);
                    return Ok(true);
                }

                let project_name = words[2..].join(" ");
                if command == "reindex" {
                    let project = sqlx::query(
                        "SELECT id, root_path FROM brain.projects WHERE LOWER(name) = LOWER($1)",
                    )
                    .bind(&project_name)
                    .fetch_optional(&self.pool)
                    .await?;

                    let Some(project) = project else {
                        println!("Proiectul '{}' nu există.", project_name);
                        return Ok(true);
                    };
                    let project_id: i64 = project.get("id");
                    let root_path: Option<String> = project.get("root_path");
                    let root_path = match root_path {
                        Some(path) => path,
                        None => {
                            let rows = sqlx::query(
                                "SELECT path FROM brain.source_files WHERE project_id = $1",
                            )
                            .bind(project_id)
                            .fetch_all(&self.pool)
                            .await?;
                            let paths: Vec<String> =
                                rows.iter().map(|row| row.get("path")).collect();
                            let Some(root_path) = infer_root_path(&paths) else {
                                println!(
                                    "Proiectul '{}' nu are directorul sursă configurat.",
                                    project_name
                                );
                                return Ok(true);
                            };
                            let root_path = root_path.to_string_lossy().into_owned();
                            sqlx::query("UPDATE brain.projects SET root_path = $1 WHERE id = $2")
                                .bind(&root_path)
                                .bind(project_id)
                                .execute(&self.pool)
                                .await?;
                            root_path
                        }
                    };

                    println!("Reindexez proiectul '{}' din {}...", project_name, root_path);
                    CodeIndexer::new(self.pool.clone())
                        .reindex_and_store(project_id, &root_path)
                        .await?;
                    println!("Reindexarea proiectului '{}' s-a încheiat.", project_name);
                } else {
                    let deleted_id = sqlx::query_scalar::<_, i64>(
                        "DELETE FROM brain.projects WHERE LOWER(name) = LOWER($1) RETURNING id",
                    )
                    .bind(&project_name)
                    .fetch_optional(&self.pool)
                    .await?;
                    if let Some(deleted_id) = deleted_id {
                        println!(
                            "Proiectul '{}' și datele sale au fost șterse.",
                            project_name
                        );
                        if deleted_id == *active_project_id {
                            println!("A fost șters și proiectul chatului curent.");
                        }
                    } else {
                        println!("Proiectul '{}' nu există.", project_name);
                    }
                }

                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// Pornește bucla interactivă REPL
    pub async fn start_repl(&self, mut project_id: i64) -> Result<(), Box<dyn Error>> {
        let mut line_editor = DefaultEditor::new()?;
        self.ensure_chat_history_table().await?;
        let mut conversation_history = self.load_chat_history(project_id).await?;
        if !conversation_history.is_empty() {
            println!(
                "Am încărcat istoricul recent al proiectului ({} mesaje).",
                conversation_history.len()
            );
        }

        println!("\n========================================================");
        println!("  🧠 Cortex Brain AI REPL (Tastați 'help' pentru comenzi, 'exit' pentru ieșire)");
        println!("========================================================\n");

        loop {
            let input = match line_editor.readline("Cortex > ") {
                Ok(input) => input,
                Err(ReadlineError::Interrupted) => {
                    println!();
                    continue;
                }
                Err(ReadlineError::Eof) => {
                    println!("La revedere!");
                    break;
                }
                Err(error) => return Err(error.into()),
            };
            let query = input.trim();

            if query.eq_ignore_ascii_case("exit") || query.is_empty() {
                println!("La revedere!");
                break;
            }
            line_editor.add_history_entry(query)?;

            let previous_project_id = project_id;
            match self.handle_command(query, &mut project_id).await {
                Ok(true) => {
                    if project_id != previous_project_id {
                        conversation_history = self.load_chat_history(project_id).await?;
                    }
                    continue;
                }
                Ok(false) => {}
                Err(error) => {
                    eprintln!("Eroare la executarea comenzii: {}", error);
                    continue;
                }
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
            match self
                .generate_answer(&full_prompt, &conversation_history)
                .await
            {
                Ok(answer) => {
                    println!("\n🤖 RĂSPUNS CORTEX:\n{}\n\n", answer);
                    remember_exchange(&mut conversation_history, query, &answer);
                    if let Err(error) = self
                        .save_chat_exchange(project_id, query, &answer)
                        .await
                    {
                        eprintln!("Eroare la salvarea istoricului conversației: {}", error);
                    }
                }
                Err(e) => eprintln!("Eroare la generarea răspunsului: {}", e),
            }
        }

        Ok(())
    }
}

fn remember_exchange(history: &mut Vec<ChatMessage>, user_prompt: &str, assistant_answer: &str) {
    history.push(ChatMessage {
        role: "user".to_string(),
        content: user_prompt.to_string(),
    });
    history.push(ChatMessage {
        role: "assistant".to_string(),
        content: assistant_answer.to_string(),
    });

    if history.len() > MAX_HISTORY_MESSAGES {
        let excess = history.len() - MAX_HISTORY_MESSAGES;
        history.drain(0..excess);
    }
}

fn infer_root_path(paths: &[String]) -> Option<PathBuf> {
    let mut root = Path::new(paths.first()?).parent()?.to_path_buf();
    for path in &paths[1..] {
        let parent = Path::new(path).parent()?;
        while !parent.starts_with(&root) {
            root = root.parent()?.to_path_buf();
        }
    }
    Some(root)
}

#[cfg(test)]
mod tests {
    use super::{infer_root_path, remember_exchange, MAX_HISTORY_MESSAGES};

    #[test]
    fn infers_common_project_root_from_indexed_file_paths() {
        let project_root = std::env::temp_dir().join("cortex-project").join("winapi");
        let paths = vec![
            project_root
                .join("src")
                .join("entry.cpp")
                .to_string_lossy()
                .into_owned(),
            project_root
                .join("include")
                .join("api.hpp")
                .to_string_lossy()
                .into_owned(),
        ];

        assert_eq!(
            infer_root_path(&paths).as_deref(),
            Some(project_root.as_path())
        );
    }

    #[test]
    fn returns_no_root_for_an_empty_file_list() {
        assert!(infer_root_path(&[]).is_none());
    }

    #[test]
    fn keeps_only_the_most_recent_complete_exchanges() {
        let mut history = Vec::new();
        for index in 0..=MAX_HISTORY_MESSAGES / 2 {
            remember_exchange(&mut history, &format!("prompt {index}"), &format!("answer {index}"));
        }

        assert_eq!(history.len(), MAX_HISTORY_MESSAGES);
        assert_eq!(history[0].role, "user");
        assert_eq!(history[0].content, "prompt 1");
        assert_eq!(history[1].role, "assistant");
        assert_eq!(history[1].content, "answer 1");
        assert_eq!(history.last().unwrap().content, "answer 10");
    }
}