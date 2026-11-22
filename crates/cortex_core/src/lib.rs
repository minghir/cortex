pub mod config;
pub mod indexer;
pub mod rag_chat;
pub mod types;

use cortex_parser::parser_for_project_type;
use indexer::{CodeIndexer, IndexStats};
use rag_chat::{ApiChatReply, RagChat};
use sqlx::{postgres::PgPoolOptions, PgPool, Row};
use std::path::Path;
use types::{ChatHistoryEntryDto, ProjectDto, RemoteFilePayload};

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Clone)]
pub struct CortexEngine {
    pool: PgPool,
}

impl CortexEngine {
    /// Returnează lista tuturor proiectelor înregistrate în baza de date.
    pub async fn list_projects(&self) -> Result<Vec<ProjectDto>, BoxError> {
        let rows = sqlx::query(
            "SELECT id, name, project_type, root_path, git_remote FROM brain.projects ORDER BY id DESC",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.iter().map(row_to_project).collect())
    }

    pub async fn update_project_root(
        &self,
        project_id: i64,
        root_path: &Path,
    ) -> Result<ProjectDto, BoxError> {
        let canon = std::fs::canonicalize(root_path)
            .unwrap_or_else(|_| root_path.to_path_buf())
            .to_string_lossy()
            .into_owned();

        let row = sqlx::query(
            r#"
            UPDATE brain.projects
            SET root_path = $1
            WHERE id = $2
            RETURNING id, name, project_type, root_path, git_remote
            "#,
        )
        .bind(&canon)
        .bind(project_id)
        .fetch_one(&self.pool)
        .await?;

        Ok(row_to_project(&row))
    }

        /// Încarcă ultimele `limit` mesaje din istoricul conversației unui proiect (în ordine cronologică).
        pub async fn load_history(
            &self,
            project_id: i64,
            limit: i64,
        ) -> Result<Vec<ChatHistoryEntryDto>, Box<dyn std::error::Error + Send + Sync>> {
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
            .bind(limit)
            .fetch_all(&self.pool)
            .await?;

            let mut messages: Vec<ChatHistoryEntryDto> = rows
                .into_iter()
                .filter_map(|row| {
                    let role: String = row.get("role");
                    let content: String = row.get("content");
                    if content.is_empty() {
                        None
                    } else {
                        Some(ChatHistoryEntryDto { role, content })
                    }
                })
                .collect();

            messages.reverse();
            Ok(messages)
        }

        /// Șterge tot istoricul conversației pentru proiectul specificat.
        pub async fn clear_history(
            &self,
            project_id: i64,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            sqlx::query("DELETE FROM brain.chat_history WHERE project_id = $1")
                .bind(project_id)
                .execute(&self.pool)
                .await?;

            Ok(())
    }
    

    /// Inițializează conexiunea PostgreSQL (citește DATABASE_URL din .env dacă nu e pasat explicit)
    pub async fn connect(database_url: Option<&str>) -> Result<Self, BoxError> {
        let _ = dotenvy::dotenv();
        let url = match database_url {
            Some(u) if !u.trim().is_empty() => u.to_string(),
            _ => std::env::var("DATABASE_URL")
                .map_err(|_| "DATABASE_URL lipsește din mediu sau .env")?,
        };

        let pool = PgPoolOptions::new()
            .max_connections(8)
            .connect(&url)
            .await?;

        Ok(Self { pool })
    }

    pub fn from_pool(pool: PgPool) -> Self {
        Self { pool }
    }

    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Leagă sau creează un proiect pe baza folderului local (`root_path`)
    pub async fn resolve_or_create_project(
        &self,
        project_id: Option<i64>,
        name: Option<&str>,
        root_path: Option<&Path>,
        git_remote: Option<&str>,
        project_type: Option<&str>,
        create_if_missing: bool,
    ) -> Result<Option<ProjectDto>, BoxError> {
        let canon_root = root_path
            .and_then(|p| std::fs::canonicalize(p).ok().or_else(|| Some(p.to_path_buf())))
            .map(|p| p.to_string_lossy().into_owned());

        if let Some(pid) = project_id {
            if let Some(row) = sqlx::query(
                "SELECT id, name, project_type, root_path, git_remote FROM brain.projects WHERE id = $1",
            )
            .bind(pid)
            .fetch_optional(&self.pool)
            .await?
            {
                return Ok(Some(row_to_project(&row)));
            }
        }

        if let Some(ref rp) = canon_root {
            if let Some(row) = sqlx::query(
                "SELECT id, name, project_type, root_path, git_remote FROM brain.projects WHERE root_path = $1 LIMIT 1",
            )
            .bind(rp)
            .fetch_optional(&self.pool)
            .await?
            {
                return Ok(Some(row_to_project(&row)));
            }
        }

        if let Some(proj_name) = name.map(str::trim).filter(|s| !s.is_empty()) {
            if let Some(row) = sqlx::query(
                "SELECT id, name, project_type, root_path, git_remote FROM brain.projects WHERE LOWER(name) = LOWER($1) LIMIT 1",
            )
            .bind(proj_name)
            .fetch_optional(&self.pool)
            .await?
            {
                return Ok(Some(row_to_project(&row)));
            }

            if create_if_missing {
                let ptype = project_type.unwrap_or("code");
                let inserted = sqlx::query(
                    r#"
                    INSERT INTO brain.projects (name, project_type, root_path, git_remote)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id, name, project_type, root_path, git_remote
                    "#,
                )
                .bind(proj_name)
                .bind(ptype)
                .bind(&canon_root)
                .bind(git_remote)
                .fetch_one(&self.pool)
                .await?;

                return Ok(Some(row_to_project(&inserted)));
            }
        }

        Ok(None)
    }

    /// Indexează incremental întregul director al proiectului de pe HDD
    pub async fn index_project_dir(&self, project_id: i64) -> Result<(), BoxError> {
        let row = sqlx::query("SELECT project_type, root_path FROM brain.projects WHERE id = $1")
            .bind(project_id)
            .fetch_one(&self.pool)
            .await?;

        let project_type: String = row.get("project_type");
        let root_path: Option<String> = row.get("root_path");
        let root_path = root_path.ok_or("Proiectul nu are root_path configurat")?;

        let parser = parser_for_project_type(&project_type);
        CodeIndexer::new(self.pool.clone())
            .index_and_store(project_id, &root_path, parser.as_ref())
            .await
    }

    /// Indexează incremental unul sau mai multe fișiere salvate în RED (`Ctrl+S`)
    pub async fn index_files(
        &self,
        project_id: i64,
        files: &[RemoteFilePayload],
    ) -> Result<IndexStats, BoxError> {
        let project_type: String = sqlx::query_scalar(
            "SELECT project_type FROM brain.projects WHERE id = $1",
        )
        .bind(project_id)
        .fetch_optional(&self.pool)
        .await?
        .unwrap_or_else(|| "code".to_string());

        CodeIndexer::new(self.pool.clone())
            .index_remote_files(project_id, files, &project_type)
            .await
    }

    /// Întreabă AI-ul (citind automat fișierele live de pe HDD + contextul RAG)
    pub async fn ask(
        &self,
        project_id: i64,
        query: &str,
        provider: &str,
        ollama_model: Option<String>,
        active_file: Option<RemoteFilePayload>,
        mentioned_files: Vec<RemoteFilePayload>,
    ) -> Result<ApiChatReply, BoxError> {
        RagChat::new(self.pool.clone())
            .ask_from_api(
                project_id,
                query,
                provider,
                ollama_model,
                active_file,
                mentioned_files,
            )
            .await
    }
}

fn row_to_project(row: &sqlx::postgres::PgRow) -> ProjectDto {
    ProjectDto {
        id: row.get("id"),
        name: row.get("name"),
        project_type: row.get("project_type"),
        root_path: row.try_get("root_path").ok().flatten(),
        git_remote: row.try_get("git_remote").ok().flatten(),
    }
}