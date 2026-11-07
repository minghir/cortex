mod indexer;
mod rag_chat;

use indexer::CodeIndexer;
use rag_chat::RagChat;
use sqlx::postgres::PgPoolOptions;
use sqlx::Row;
use std::error::Error;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Cortex Brain Initializer ===");

    let db_url = "postgres://postgres:alcatraz@localhost:5433/cortex";
    println!("[1/3] Încerc conectarea la DB...");

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .acquire_timeout(Duration::from_secs(5))
        .connect(db_url)
        .await?;

    println!("[2/3] Conectat la PostgreSQL!");

    // 1. Căutăm mai întâi dacă proiectul există deja după nume
    let project_name = "lib_winui";
    let existing_project = sqlx::query("SELECT id FROM brain.projects WHERE name = $1")
        .bind(project_name)
        .fetch_optional(&pool)
        .await?;

    let project_id: i64 = match existing_project {
        Some(row) => row.get("id"),
        None => {
            println!("📌 Proiect nou detectat. Îl creăm în DB...");
            let new_row = sqlx::query(
                r#"
                INSERT INTO brain.projects (name, description, tech_stack)
                VALUES ($1, $2, $3)
                RETURNING id
                "#,
            )
            .bind(project_name)
            .bind("Librărie UI C++ WinUI")
            .bind(serde_json::json!(["C++", "WinUI"]))
            .fetch_one(&pool)
            .await?;
            new_row.get("id")
        }
    };

    println!("📌 Proiect selectat: '{}' (ID: {})", project_name, project_id);

    // 2. Verificăm numărul de fișiere indexate pentru acest ID
    let count_row = sqlx::query("SELECT COUNT(*) as count FROM brain.source_files WHERE project_id = $1")
        .bind(project_id)
        .fetch_one(&pool)
        .await?;
    
    let files_count: i64 = count_row.get("count");
    let target_cpp_dir = "D:\\Programming\\lib_winui";

    if files_count > 0 {
        println!("⚡ Proiectul este deja indexat (găsite {} fișiere în DB). Sare peste scanare!", files_count);
    } else {
        println!("[3/3] Pornim scanarea inițială din: {}", target_cpp_dir);
        let indexer = CodeIndexer::new(pool.clone());
        indexer.process_and_store(project_id, target_cpp_dir).await?;
        println!("Indexare finalizată!");
    }

    // 3. Pornire REPL Chat
    let chat = RagChat::new(pool);
    chat.start_repl(project_id).await?;

    Ok(())
}