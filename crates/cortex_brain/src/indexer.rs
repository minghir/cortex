use crate::config::ollama_url;
use cortex_parser::CppParser;
use pgvector::Vector;
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Row};
use std::error::Error;
use std::io;
use std::path::Path;

#[derive(Serialize)]
struct OllamaEmbedReq {
    model: String,
    prompt: String,
}

#[derive(Deserialize)]
struct OllamaEmbedRes {
    embedding: Vec<f32>,
}

pub struct CodeIndexer {
    pool: PgPool,
    client: reqwest::Client,
}

impl CodeIndexer {
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool,
            client: reqwest::Client::new(),
        }
    }

    async fn get_embedding(&self, text: &str) -> Result<Vec<f32>, Box<dyn Error>> {
        let req = OllamaEmbedReq {
            model: "nomic-embed-text".to_string(),
            prompt: text.to_string(),
        };

        println!("   └─► Solicit embedding de la Ollama (nomic-embed-text)...");
        let res = self
            .client
            .post(ollama_url("/api/embeddings"))
            .json(&req)
            .send()
            .await?
            .json::<OllamaEmbedRes>()
            .await?;

        Ok(res.embedding)
    }

    pub async fn process_and_store(&self, project_id: i64, dir_path: &str) -> Result<(), Box<dyn Error>> {
        self.process_and_store_internal(project_id, dir_path, false).await
    }

    pub async fn reindex_and_store(
        &self,
        project_id: i64,
        dir_path: &str,
    ) -> Result<(), Box<dyn Error>> {
        self.process_and_store_internal(project_id, dir_path, true).await
    }

    async fn process_and_store_internal(
        &self,
        project_id: i64,
        dir_path: &str,
        reindex: bool,
    ) -> Result<(), Box<dyn Error>> {
        if !Path::new(dir_path).is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Directorul proiectului nu există: {}", dir_path),
            )
            .into());
        }

        println!("   ├─ Parsez structura fișierelor C++ cu cortex_parser...");
        let parsed_files = CppParser::parse_directory(dir_path)?;
        println!("   ├─ Găsite {} fișiere C++.", parsed_files.len());

        if reindex && parsed_files.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Nu s-au găsit fișiere C++; indexul existent a fost păstrat.",
            )
            .into());
        }

        let mut tx = self.pool.begin().await?;
        sqlx::query("UPDATE brain.projects SET root_path = $1 WHERE id = $2")
            .bind(dir_path)
            .bind(project_id)
            .execute(&mut *tx)
            .await?;

        if reindex {
            sqlx::query("DELETE FROM brain.memories WHERE project_id = $1")
                .bind(project_id)
                .execute(&mut *tx)
                .await?;
            sqlx::query("DELETE FROM brain.source_files WHERE project_id = $1")
                .bind(project_id)
                .execute(&mut *tx)
                .await?;
        }

        for (idx, file) in parsed_files.iter().enumerate() {
            println!("\n   [{}/{}] Procesez fișierul: {}", idx + 1, parsed_files.len(), file.path);
            let summary = format!("Fișier {} cu {} clase", file.path, file.classes.len());

            println!("   ├─ Salvez fișierul în brain.source_files...");
            let source_file_row = sqlx::query(
                r#"
                INSERT INTO brain.source_files (project_id, path, summary, language)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (project_id, path) 
                DO UPDATE SET summary = EXCLUDED.summary
                RETURNING id
                "#,
            )
            .bind(project_id)
            .bind(&file.path)
            .bind(&summary)
            .bind(&file.extension)
            .fetch_one(&mut *tx)
            .await?;

            let file_id: i64 = source_file_row.get("id");

            let content_chunks = chunk_content(&file.content, 1200);
            for (chunk_index, chunk) in content_chunks.iter().enumerate() {
                let content_type = if file.extension == "md" {
                    "documentation"
                } else {
                    "source_code"
                };
                let mem_text = format!(
                    "Fișier: {}\nFragment {}/{}:\n{}",
                    file.path,
                    chunk_index + 1,
                    content_chunks.len(),
                    chunk
                );
                let embedding = self.get_embedding(&mem_text).await?;
                sqlx::query(
                    r#"
                    INSERT INTO brain.memories
                        (project_id, content, content_type, source_table, source_id, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    "#,
                )
                .bind(project_id)
                .bind(&mem_text)
                .bind(content_type)
                .bind("source_files")
                .bind(file_id)
                .bind(Vector::from(embedding))
                .execute(&mut *tx)
                .await?;
            }

            for cls in &file.classes {
                println!("   ├─ Salvez clasa: {}", cls.name);
                let class_row = sqlx::query(
                    r#"
                    INSERT INTO brain.classes (file_id, class_name, description, class_definition)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id
                    "#,
                )
                .bind(file_id)
                .bind(&cls.name)
                .bind(format!("Clasa {}", cls.name))
                .bind(&cls.definition)
                .fetch_one(&mut *tx)
                .await?;

                let class_id: i64 = class_row.get("id");

                for func in &cls.functions {
                    println!("   │  ├─ Funcție: {}", func.name);
                    let fn_row = sqlx::query(
                        r#"
                        INSERT INTO brain.functions (class_id, function_name, signature, description)
                        VALUES ($1, $2, $3, $4)
                        RETURNING id
                        "#,
                    )
                    .bind(class_id)
                    .bind(&func.name)
                    .bind(&func.signature)
                    .bind(format!("Funcția {}", func.name))
                    .fetch_one(&mut *tx)
                    .await?;

                    let fn_id: i64 = fn_row.get("id");

                    let mem_text = format!("Clasă: {}\nFuncție: {}\nSemnătură: {}", cls.name, func.name, func.signature);
                    let embedding = self.get_embedding(&mem_text).await?;
                    let vector = Vector::from(embedding);
                    sqlx::query(
                        r#"
                        INSERT INTO brain.memories (project_id, content, content_type, source_table, source_id, embedding)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        "#,
                    )
                    .bind(project_id)
                    .bind(&mem_text)
                    .bind("function")
                    .bind("functions")
                    .bind(fn_id)
                    .bind(vector)
                    .execute(&mut *tx)
                    .await?;
                    println!("   │  └─► Salvat în brain.memories!");
                }

            }
        }

        tx.commit().await?;
        Ok(())
    }
}

fn chunk_content(content: &str, chunk_size: usize) -> Vec<String> {
    let chars: Vec<char> = content.chars().collect();
    chars
        .chunks(chunk_size)
        .map(|chunk| chunk.iter().collect())
        .collect()
}