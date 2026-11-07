use cortex_parser::CppParser;
use pgvector::Vector;
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Row};
use std::error::Error;

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
            .post("http://localhost:11434/api/embeddings")
            .json(&req)
            .send()
            .await?
            .json::<OllamaEmbedRes>()
            .await?;

        Ok(res.embedding)
    }

    pub async fn process_and_store(&self, project_id: i64, dir_path: &str) -> Result<(), Box<dyn Error>> {
        println!("   ├─ Parsez structura fișierelor C++ cu cortex_parser...");
        let parsed_files = CppParser::parse_directory(dir_path)?;
        println!("   ├─ Găsite {} fișiere C++.", parsed_files.len());

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
            .fetch_one(&self.pool)
            .await?;

            let file_id: i64 = source_file_row.get("id");

            for cls in &file.classes {
                println!("   ├─ Salvez clasa: {}", cls.name);
                let class_row = sqlx::query(
                    r#"
                    INSERT INTO brain.classes (file_id, class_name, description)
                    VALUES ($1, $2, $3)
                    RETURNING id
                    "#,
                )
                .bind(file_id)
                .bind(&cls.name)
                .bind(format!("Clasa {}", cls.name))
                .fetch_one(&self.pool)
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
                    .fetch_one(&self.pool)
                    .await?;

                    let fn_id: i64 = fn_row.get("id");

                    let mem_text = format!("Clasă: {}\nFuncție: {}\nSemnătură: {}", cls.name, func.name, func.signature);
                    match self.get_embedding(&mem_text).await {
                        Ok(emb) => {
                            let vector = Vector::from(emb);
                            let _ = sqlx::query(
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
                            .execute(&self.pool)
                            .await?;
                            println!("   │  └─► Salvat în brain.memories!");
                        }
                        Err(err) => eprintln!("   │  └─❌ Eroare embedding Ollama: {}", err),
                    }
                }
            }
        }

        Ok(())
    }
}