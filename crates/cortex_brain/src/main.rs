use cortex_core::{config, indexer, rag_chat};
mod server;


use cortex_parser::parser_for_project_type;
use indexer::CodeIndexer;
use rag_chat::RagChat;
use sqlx::postgres::PgPoolOptions;
use sqlx::Row;
use std::error::Error;
use std::io;
use std::path::Path;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
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
        .max_connections(10)
        .acquire_timeout(Duration::from_secs(5))
        .connect(&db_url)
        .await?;

    println!("[2/3] Conectat la PostgreSQL!");

    // Mod Server HTTP pentru conectare din RED (local sau prin Cloudflare Tunnel / Tailscale)
    let serve_mode = std::env::args().any(|arg| arg == "--serve" || arg == "serve")
        || std::env::var("CORTEX_SERVER_MODE").as_deref() == Ok("1");

    if serve_mode {
        println!("[3/3] Pornire în modul Cortex HTTP Server...");
        server::start_server(pool).await?;
        return Ok(());
    }

    // --- Modul REPL existent (neschimbat) ---
    let target_dir =
        std::env::var("PROJECT_SOURCE_DIR").unwrap_or_else(|_| "../nanovi".to_string());

    let project_name = match std::env::var("PROJECT_NAME") {
        Ok(name) => name,
        Err(_) => Path::new(&target_dir)
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

    let default_project_type =
        std::env::var("PROJECT_TYPE").unwrap_or_else(|_| "cpp_code".to_string());

    let existing_project =
        sqlx::query("SELECT id, project_type FROM brain.projects WHERE name = $1")
            .bind(&project_name)
            .fetch_optional(&pool)
            .await?;

    let (project_id, project_type): (i64, String) = match existing_project {
        Some(row) => (row.get("id"), row.get("project_type")),
        None => {
            println!(
                "📌 Proiect nou detectat (tip: {}). Îl creăm în DB...",
                default_project_type
            );
            let new_row = sqlx::query(
                r#"
                INSERT INTO brain.projects (name, project_type, root_path)
                VALUES ($1, $2, $3)
                RETURNING id, project_type
                "#,
            )
            .bind(&project_name)
            .bind(&default_project_type)
            .bind(&target_dir)
            .fetch_one(&pool)
            .await?;
            (new_row.get("id"), new_row.get("project_type"))
        }
    };

    sqlx::query("UPDATE brain.projects SET root_path = COALESCE(root_path, $1) WHERE id = $2")
        .bind(&target_dir)
        .bind(project_id)
        .execute(&pool)
        .await?;

    println!(
        "📌 Proiect selectat: '{}' [tip: {}] (ID: {})",
        project_name, project_type, project_id
    );

    let count_row =
        sqlx::query("SELECT COUNT(*) as count FROM brain.source_files WHERE project_id = $1")
            .bind(project_id)
            .fetch_one(&pool)
            .await?;

    let files_count: i64 = count_row.get("count");
    if files_count > 0 {
        println!(
            "⚡ Proiectul este deja indexat (găsite {} fișiere în DB). Sare peste scanare!",
            files_count
        );
    } else {
        println!("[3/3] Pornim scanarea inițială din: {}", target_dir);
        let indexer = CodeIndexer::new(pool.clone());
        let parser = parser_for_project_type(&project_type);

        indexer
            .index_and_store(project_id, &target_dir, parser.as_ref())
            .await?;
        println!("✅ Indexare finalizată!");
    }

    let chat = RagChat::new(pool);
    chat.start_repl(project_id).await?;

    Ok(())
}