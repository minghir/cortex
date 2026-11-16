mod config;
mod indexer;
mod rag_chat;

use indexer::CodeIndexer;
use rag_chat::RagChat;
use sqlx::postgres::PgPoolOptions;
use sqlx::Row;
use std::error::Error;
use std::io;
use std::path::Path;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Cortex Brain Initializer ===");

    if let Err(error) = dotenvy::dotenv() {
        if !error.not_found() {
            return Err(error.into());
        }
    }

    let db_url = std::env::var("DATABASE_URL").map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Configurează DATABASE_URL în .env (vezi .env.example) sau ca variabilă de mediu.",
        )
    })?;
    println!("[1/3] Încerc conectarea la DB...");

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .acquire_timeout(Duration::from_secs(5))
        .connect(&db_url)
        .await?;

    println!("[2/3] Conectat la PostgreSQL!");

    // 1. Căutăm mai întâi dacă proiectul există deja după nume
    let target_cpp_dir = std::env::var("PROJECT_SOURCE_DIR")
        .unwrap_or_else(|_| "../nanovi".to_string());
    let project_name = match std::env::var("PROJECT_NAME") {
        Ok(name) => name,
        Err(_) => Path::new(&target_cpp_dir)
            .file_name()
            .and_then(|name| name.to_str())
            .map(str::to_string)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Setează PROJECT_NAME sau folosește un director cu nume valid.",
                )
            })?,
    };
    let existing_project = sqlx::query("SELECT id FROM brain.projects WHERE name = $1")
        .bind(&project_name)
        .fetch_optional(&pool)
        .await?;

    let project_id: i64 = match existing_project {
        Some(row) => row.get("id"),
        None => {
            println!("📌 Proiect nou detectat. Îl creăm în DB...");
            let new_row = sqlx::query(
                r#"
                INSERT INTO brain.projects (name, root_path)
                VALUES ($1, $2)
                RETURNING id
                "#,
            )
            .bind(&project_name)
            .bind(&target_cpp_dir)
            .fetch_one(&pool)
            .await?;
            new_row.get("id")
        }
    };

    sqlx::query("UPDATE brain.projects SET root_path = COALESCE(root_path, $1) WHERE id = $2")
        .bind(&target_cpp_dir)
        .bind(project_id)
        .execute(&pool)
        .await?;

    println!("📌 Proiect selectat: '{}' (ID: {})", project_name, project_id);

    // 2. Verificăm numărul de fișiere indexate pentru acest ID
    let count_row = sqlx::query("SELECT COUNT(*) as count FROM brain.source_files WHERE project_id = $1")
        .bind(project_id)
        .fetch_one(&pool)
        .await?;
    
    let files_count: i64 = count_row.get("count");
    if files_count > 0 {
        println!("⚡ Proiectul este deja indexat (găsite {} fișiere în DB). Sare peste scanare!", files_count);
    } else {
        println!("[3/3] Pornim scanarea inițială din: {}", target_cpp_dir);
        let indexer = CodeIndexer::new(pool.clone());
        indexer
            .process_and_store(project_id, &target_cpp_dir)
            .await?;
        println!("Indexare finalizată!");
    }

    // 3. Pornire REPL Chat
    let chat = RagChat::new(pool);
    chat.start_repl(project_id).await?;

    Ok(())
}