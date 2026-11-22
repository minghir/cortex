use std::error::Error;
use std::sync::Arc;

use axum::{
    extract::{Query, Request, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{Html, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Row};

use crate::indexer::CodeIndexer;
use crate::rag_chat::RagChat;

pub use cortex_core::types::{RemoteFilePayload, TokenUsageDto};

pub struct AppState {
    pub pool: PgPool,
    pub api_key: Option<String>,
}

// --- DTO-uri pentru API ---

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub service: &'static str,
}

#[derive(Serialize)]
pub struct ProjectDto {
    pub id: i64,
    pub name: String,
    pub project_type: String,
    pub root_path: Option<String>,
}

#[derive(Deserialize)]
pub struct EnsureProjectRequest {
    pub name: String,
    pub project_type: Option<String>,
    pub root_path: Option<String>,
}

#[derive(Deserialize)]
pub struct HistoryQuery {
    pub project_id: i64,
    pub limit: Option<i64>,
}

#[derive(Serialize)]
pub struct HistoryMessageDto {
    pub role: String,
    pub content: String,
}


#[derive(Serialize)]
pub struct OllamaModelDto {
    pub name: String,
    pub size_bytes: Option<u64>,
    pub parameter_size: Option<String>,
}

#[derive(Serialize)]
pub struct OllamaModelsResponse {
    pub models: Vec<OllamaModelDto>,
}

#[derive(Deserialize)]
pub struct ChatRequest {
    pub project_id: i64,
    pub message: String,
    /// "ollama" sau "google"
    pub llm_provider: Option<String>,
    /// Modelul Ollama selectat (opțional)
    pub ollama_model: Option<String>,
    /// Fișierul deschis în tab-ul activ din RED
    pub active_file: Option<RemoteFilePayload>,
    /// Fișierele @fisier rezolvate local de RED
    #[serde(default)]
    pub mentioned_files: Vec<RemoteFilePayload>,
}


#[derive(Serialize)]
pub struct ChatResponse {
    pub answer: String,
    pub llm_provider: String,
    pub usage: TokenUsageDto,
}

#[derive(Deserialize)]
pub struct IndexFilesRequest {
    pub project_id: i64,
    pub files: Vec<RemoteFilePayload>,
}

#[derive(Serialize)]
pub struct IndexFilesResponse {
    pub indexed_count: usize,
    pub skipped_unchanged: usize,
}

#[derive(Deserialize)]
pub struct ResolveProjectRequest {
    pub project_id: Option<i64>,
    pub name: Option<String>,
    pub root_path: Option<String>,
    pub git_remote: Option<String>,
    pub project_type: Option<String>,
    #[serde(default)]
    pub create_if_missing: bool,
}

#[derive(Serialize)]
pub struct ResolveProjectResponse {
    pub project: Option<ProjectDto>,
    pub matched_by: Option<String>,
}


fn normalize_path_for_match(raw: &str) -> String {
    raw.trim()
        .replace('\\', "/")
        .trim_end_matches('/')
        .to_ascii_lowercase()
}

fn row_to_project_dto(row: sqlx::postgres::PgRow) -> ProjectDto {
    ProjectDto {
        id: row.get("id"),
        name: row.get("name"),
        project_type: row.get("project_type"),
        root_path: row.try_get("root_path").ok().flatten(),
    }
}

async fn resolve_project(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ResolveProjectRequest>,
) -> Result<Json<ResolveProjectResponse>, (StatusCode, String)> {
    // 1. Potrivire directă după project_id (ex. din .cortex.toml)
    if let Some(pid) = req.project_id {
        if let Some(row) = sqlx::query(
            "SELECT id, name, project_type, root_path FROM brain.projects WHERE id = $1",
        )
        .bind(pid)
        .fetch_optional(&state.pool)
        .await
        .map_err(internal_err)?
        {
            return Ok(Json(ResolveProjectResponse {
                project: Some(row_to_project_dto(row)),
                matched_by: Some("project_id".to_string()),
            }));
        }
    }

    // 2. Potrivire după root_path normalizat (dacă există în DB)
    if let Some(ref root) = req.root_path {
        let norm_root = normalize_path_for_match(root);
        if !norm_root.is_empty() {
            let rows = sqlx::query(
                "SELECT id, name, project_type, root_path FROM brain.projects WHERE root_path IS NOT NULL",
            )
            .fetch_all(&state.pool)
            .await
            .map_err(internal_err)?;

            if let Some(row) = rows.into_iter().find(|r| {
                r.try_get::<Option<String>, _>("root_path")
                    .ok()
                    .flatten()
                    .map(|p| normalize_path_for_match(&p) == norm_root)
                    .unwrap_or(false)
            }) {
                return Ok(Json(ResolveProjectResponse {
                    project: Some(row_to_project_dto(row)),
                    matched_by: Some("root_path".to_string()),
                }));
            }
        }
    }

    // 3. Potrivire după nume proiect (case-insensitive)
    if let Some(ref name) = req.name {
        let trimmed = name.trim();
        if !trimmed.is_empty() {
            if let Some(row) = sqlx::query(
                "SELECT id, name, project_type, root_path FROM brain.projects WHERE LOWER(name) = LOWER($1) ORDER BY id ASC LIMIT 1",
            )
            .bind(trimmed)
            .fetch_optional(&state.pool)
            .await
            .map_err(internal_err)?
            {
                return Ok(Json(ResolveProjectResponse {
                    project: Some(row_to_project_dto(row)),
                    matched_by: Some("name".to_string()),
                }));
            }
        }
    }

    // 4. Opțional: creează proiectul dacă nu există și s-a cerut explicit
    if req.create_if_missing {
        let fallback_name = req
            .name
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .or_else(|| {
                req.root_path.as_deref().and_then(|p| {
                    std::path::Path::new(p)
                        .file_name()
                        .and_then(|n| n.to_str())
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                })
            });

        if let Some(proj_name) = fallback_name {
            let ptype = req
                .project_type
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .unwrap_or("cpp_code");

            let created = sqlx::query(
                r#"
                INSERT INTO brain.projects (name, project_type, root_path)
                VALUES ($1, $2, $3)
                RETURNING id, name, project_type, root_path
                "#,
            )
            .bind(proj_name)
            .bind(ptype)
            .bind(&req.root_path)
            .fetch_one(&state.pool)
            .await
            .map_err(internal_err)?;

            return Ok(Json(ResolveProjectResponse {
                project: Some(row_to_project_dto(created)),
                matched_by: Some("created".to_string()),
            }));
        }
    }

    Ok(Json(ResolveProjectResponse {
        project: None,
        matched_by: None,
    }))
}

// --- Pornire Server ---
pub async fn start_server(pool: PgPool) -> Result<(), Box<dyn Error + Send + Sync>> {
    let bind_addr = std::env::var("CORTEX_BIND").unwrap_or_else(|_| "0.0.0.0:7878".to_string());
    let api_key = std::env::var("CORTEX_API_KEY")
        .ok()
        .map(|k| k.trim().to_string())
        .filter(|k| !k.is_empty());

    if api_key.is_none() {
        eprintln!(
            "⚠️  ATENȚIE: CORTEX_API_KEY nu este setat în .env! Endpoint-urile sunt fără autentificare."
        );
    }

    let state = Arc::new(AppState { pool, api_key });

    let protected_routes = Router::new()
        .route("/api/projects", get(list_projects).post(ensure_project))
        .route("/api/projects/resolve", post(resolve_project))
        .route("/api/history", get(get_history).delete(clear_history))
        .route("/api/ollama/models", get(list_ollama_models))
        .route("/api/chat", post(handle_chat))
        .route("/api/index", post(handle_index))
        .layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            auth_middleware,
        ));

    let app = Router::new()
        .route("/", get(handle_ui))
        .route("/api/health", get(health))
        .merge(protected_routes)
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    println!("🚀 Cortex HTTP Server ascultă pe http://{}", bind_addr);

    axum::serve(listener, app).await?;
    Ok(())
}

// --- Middleware Bearer Token ---

async fn auth_middleware(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let Some(expected_key) = &state.api_key else {
        return Ok(next.run(request).await);
    };

    let provided = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .map(str::trim);

    if provided == Some(expected_key.as_str()) {
        Ok(next.run(request).await)
    } else {
        Err(StatusCode::UNAUTHORIZED)
    }
}

// --- Handlers ---

async fn handle_ui() -> Html<&'static str> {
    Html(TEST_UI_HTML)
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        service: "cortex_brain",
    })
}

#[derive(Deserialize)]
struct OllamaTagsResponse {
    models: Option<Vec<OllamaTagItem>>,
}

#[derive(Deserialize)]
struct OllamaTagItem {
    name: String,
    size: Option<u64>,
    details: Option<OllamaModelDetails>,
}

#[derive(Deserialize)]
struct OllamaModelDetails {
    parameter_size: Option<String>,
}

async fn list_ollama_models() -> Result<Json<OllamaModelsResponse>, (StatusCode, String)> {
    let base_url = std::env::var("OLLAMA_BASE_URL")
        .or_else(|_| std::env::var("OLLAMA_URL"))
        .unwrap_or_else(|_| "http://127.0.0.1:11434".to_string());
    let url = format!("{}/api/tags", base_url.trim_end_matches('/'));

    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(3))
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(internal_err)?;

    let resp = client.get(&url).send().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("Nu pot contacta Ollama la {url}: {e}"),
        )
    })?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err((
            StatusCode::BAD_GATEWAY,
            format!("Ollama /api/tags HTTP {status}: {body}"),
        ));
    }

    let parsed: OllamaTagsResponse = resp.json().await.map_err(internal_err)?;
    let models = parsed
        .models
        .unwrap_or_default()
        .into_iter()
        .map(|m| OllamaModelDto {
            name: m.name,
            size_bytes: m.size,
            parameter_size: m.details.and_then(|d| d.parameter_size),
        })
        .collect();

    Ok(Json(OllamaModelsResponse { models }))
}

async fn list_projects(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<ProjectDto>>, (StatusCode, String)> {
    let rows = sqlx::query(
        "SELECT id, name, project_type, root_path FROM brain.projects ORDER BY id DESC",
    )
    .fetch_all(&state.pool)
    .await
    .map_err(internal_err)?;

    let projects = rows
        .into_iter()
        .map(|row| ProjectDto {
            id: row.get("id"),
            name: row.get("name"),
            project_type: row.get("project_type"),
            root_path: row.get("root_path"),
        })
        .collect();

    Ok(Json(projects))
}

async fn ensure_project(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EnsureProjectRequest>,
) -> Result<Json<ProjectDto>, (StatusCode, String)> {
    let existing = sqlx::query(
        "SELECT id, name, project_type, root_path FROM brain.projects WHERE LOWER(name) = LOWER($1)",
    )
    .bind(&req.name)
    .fetch_optional(&state.pool)
    .await
    .map_err(internal_err)?;

    if let Some(row) = existing {
        return Ok(Json(ProjectDto {
            id: row.get("id"),
            name: row.get("name"),
            project_type: row.get("project_type"),
            root_path: row.get("root_path"),
        }));
    }

    let project_type = req.project_type.unwrap_or_else(|| "cpp_code".to_string());
    let new_row = sqlx::query(
        r#"
        INSERT INTO brain.projects (name, project_type, root_path)
        VALUES ($1, $2, $3)
        RETURNING id, name, project_type, root_path
        "#,
    )
    .bind(&req.name)
    .bind(&project_type)
    .bind(&req.root_path)
    .fetch_one(&state.pool)
    .await
    .map_err(internal_err)?;

    Ok(Json(ProjectDto {
        id: new_row.get("id"),
        name: new_row.get("name"),
        project_type: new_row.get("project_type"),
        root_path: new_row.get("root_path"),
    }))
}

async fn get_history(
    State(state): State<Arc<AppState>>,
    Query(query): Query<HistoryQuery>,
) -> Result<Json<Vec<HistoryMessageDto>>, (StatusCode, String)> {
    let limit = query.limit.unwrap_or(20).clamp(1, 100);

    let rows = sqlx::query(
        r#"
        SELECT role, content
        FROM (
            SELECT id, role, content
            FROM brain.chat_history
            WHERE project_id = $1
            ORDER BY id DESC
            LIMIT $2
        ) sub
        ORDER BY id ASC
        "#,
    )
    .bind(query.project_id)
    .bind(limit)
    .fetch_all(&state.pool)
    .await
    .map_err(internal_err)?;

    let history = rows
        .into_iter()
        .map(|row| HistoryMessageDto {
            role: row.get("role"),
            content: row.get("content"),
        })
        .collect();

    Ok(Json(history))
}

async fn clear_history(
    State(state): State<Arc<AppState>>,
    Query(query): Query<HistoryQuery>,
) -> Result<StatusCode, (StatusCode, String)> {
    sqlx::query("DELETE FROM brain.chat_history WHERE project_id = $1")
        .bind(query.project_id)
        .execute(&state.pool)
        .await
        .map_err(internal_err)?;

    Ok(StatusCode::NO_CONTENT)
}

async fn handle_chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, (StatusCode, String)> {
    let provider = req
        .llm_provider
        .unwrap_or_else(|| "google".to_string())
        .to_lowercase();

    let selected_ollama_model = req
        .ollama_model
        .as_deref()
        .map(str::trim)
        .filter(|m| !m.is_empty())
        .map(str::to_string);

    if provider == "ollama" {
        let model_label = selected_ollama_model
            .clone()
            .or_else(|| std::env::var("OLLAMA_MODEL").ok())
            .unwrap_or_else(|| "default (din config)".to_string());

        eprintln!(
            "🦙 [SERVER] /api/chat | proiect=#{} | provider=ollama | model={}",
            req.project_id, model_label
        );
    } else {
        eprintln!(
            "✨ [SERVER] /api/chat | proiect=#{} | provider={}",
            req.project_id, provider
        );
    }

    let chat = RagChat::new(state.pool.clone());

    let reply = chat
        .ask_from_api(
            req.project_id,
            &req.message,
            &provider,
            selected_ollama_model,
            req.active_file,
            req.mentioned_files,
        )
        .await
        .map_err(internal_err)?;

    Ok(Json(ChatResponse {
        answer: reply.answer,
        llm_provider: provider,
        usage: reply.usage,
    }))
}

async fn handle_index(
    State(state): State<Arc<AppState>>,
    Json(req): Json<IndexFilesRequest>,
) -> Result<Json<IndexFilesResponse>, (StatusCode, String)> {
    let row = sqlx::query("SELECT project_type FROM brain.projects WHERE id = $1")
        .bind(req.project_id)
        .fetch_optional(&state.pool)
        .await
        .map_err(internal_err)?
        .ok_or_else(|| (StatusCode::NOT_FOUND, "Proiect inexistent".to_string()))?;

    let project_type: String = row.get("project_type");
    let indexer = CodeIndexer::new(state.pool.clone());

    let stats = indexer
        .index_remote_files(req.project_id, &req.files, &project_type)
        .await
        .map_err(internal_err)?;

    Ok(Json(IndexFilesResponse {
        indexed_count: stats.indexed_count,
        skipped_unchanged: stats.skipped_unchanged,
    }))
}

fn internal_err<E: std::fmt::Display>(err: E) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, err.to_string())
}

const TEST_UI_HTML: &str = r#"<!DOCTYPE html>
<html lang="ro">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Cortex Brain — Web Test Console</title>
  <style>
    :root {
      --bg: #0f1117;
      --panel: #171a23;
      --border: #2a2f3d;
      --text: #e6e9f0;
      --muted: #8b93a7;
      --accent: #3b82f6;
      --accent-hover: #2563eb;
      --ok: #22c55e;
      --err: #ef4444;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      background: var(--bg);
      color: var(--text);
      padding: 20px;
    }
    .wrap { max-width: 1100px; margin: 0 auto; display: grid; gap: 16px; }
    header {
      display: flex; align-items: center; justify-content: space-between;
      background: var(--panel); border: 1px solid var(--border);
      padding: 12px 16px; border-radius: 8px;
    }
    .badge { font-size: 12px; padding: 4px 10px; border-radius: 999px; background: #1e293b; color: var(--muted); }
    .badge.ok { color: var(--ok); border: 1px solid rgba(34,197,94,.35); }
    .badge.err { color: var(--err); border: 1px solid rgba(239,68,68,.35); }
    .grid { display: grid; grid-template-columns: 1.4fr 1fr; gap: 16px; }
    @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }
    .card {
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 8px; padding: 16px; display: flex; flex-direction: column; gap: 12px;
    }
    h2 { margin: 0; font-size: 14px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
    .row { display: flex; gap: 10px; flex-wrap: wrap; }
    label { display: flex; flex-direction: column; gap: 6px; font-size: 12px; color: var(--muted); flex: 1; }
    select, input, textarea {
      background: #0b0d12; color: var(--text); border: 1px solid var(--border);
      border-radius: 6px; padding: 8px 10px; font-family: inherit; font-size: 13px;
    }
    textarea { min-height: 110px; resize: vertical; }
    button {
      background: var(--accent); color: #fff; border: none; border-radius: 6px;
      padding: 9px 14px; font-family: inherit; font-size: 13px; font-weight: 600;
      cursor: pointer; align-self: flex-start;
    }
    button:hover { background: var(--accent-hover); }
    button:disabled { opacity: .55; cursor: wait; }
    pre {
      margin: 0; background: #0b0d12; border: 1px solid var(--border);
      border-radius: 6px; padding: 12px; white-space: pre-wrap; word-break: break-word;
      font-size: 13px; line-height: 1.5; max-height: 420px; overflow-y: auto;
    }
    .meta { font-size: 12px; color: var(--muted); }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div><strong>🧠 Cortex Brain</strong> — Test Console</div>
      <span id="healthBadge" class="badge">Verific conexiunea...</span>
    </header>

    <div class="card">
      <div class="row">
        <label>
          Proiect activ (PostgreSQL)
          <select id="projectSelect"></select>
        </label>
        <label>
          Provider LLM
          <select id="providerSelect" onchange="onProviderChange()">
            <option value="google">Google Gemini</option>
            <option value="ollama">Ollama (Local)</option>
          </select>
        </label>
        <label id="ollamaModelWrap" style="display:none;">
          Model Ollama
          <select id="ollamaModelSelect"></select>
        </label>
        <label>
          CORTEX_API_KEY (opțional)
          <input id="apiKeyInput" type="password" placeholder="Bearer token din .env" onchange="saveKey()" />
        </label>
        <label style="flex: 0 0 auto; justify-content: flex-end;">
          &nbsp;
          <button type="button" onclick="loadProjectsAndModels()">Reîncarcă</button>
        </label>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <h2>1. RAG Chat (POST /api/chat)</h2>
        <label>
          Întrebare (poți menționa fișiere cu @nume_fisier)
          <textarea id="chatPrompt" placeholder="Ex: Cum funcționează parserul principal?"></textarea>
        </label>
        <details>
          <summary style="cursor:pointer; font-size:12px; color:var(--muted);">Atașează fișier activ (opțional)</summary>
          <div class="row" style="margin-top:8px;">
            <label>
              Cale relativă
              <input id="activePath" placeholder="ex: src/main.rs" />
            </label>
          </div>
          <label style="margin-top:8px;">
            Conținut fișier activ
            <textarea id="activeContent" style="min-height:70px;" placeholder="Cod sursă..."></textarea>
          </label>
        </details>
        <div class="row" style="align-items:center;">
          <button id="sendBtn" type="button" onclick="sendChat()">Trimite întrebarea</button>
          <span id="usageMeta" class="meta"></span>
        </div>
        <pre id="chatOutput">Răspunsul va apărea aici...</pre>
      </div>

      <div class="card">
        <h2>2. Indexare Incrementală SHA-256 (POST /api/index)</h2>
        <label>
          Cale relativă fișier
          <input id="idxPath" value="ping_test.rs" />
        </label>
        <label>
          Conținut fișier
          <textarea id="idxContent">pub fn cortex_ping() -> i32 { 42 }</textarea>
        </label>
        <button id="idxBtn" type="button" onclick="sendIndex()">Indexează fișierul</button>
        <pre id="idxOutput">Trimite de 2 ori același conținut pentru a testa skip-ul SHA-256...</pre>
      </div>
    </div>
  </div>

  <script>
    const keyInput = document.getElementById('apiKeyInput');
    keyInput.value = localStorage.getItem('cortex_api_key') || '';
    function saveKey() { localStorage.setItem('cortex_api_key', keyInput.value.trim()); }
    function authHeaders(extra = {}) {
      const k = keyInput.value.trim();
      return k ? { ...extra, 'Authorization': 'Bearer ' + k } : extra;
    }

    function onProviderChange() {
      const p = document.getElementById('providerSelect').value;
      const wrap = document.getElementById('ollamaModelWrap');
      wrap.style.display = p === 'ollama' ? 'flex' : 'none';
      if (p === 'ollama') {
        loadOllamaModels();
      }
    }

    async function checkHealth() {
      const badge = document.getElementById('healthBadge');
      try {
        const res = await fetch('/api/health');
        const data = await res.json();
        badge.textContent = `ONLINE (${data.service})`;
        badge.className = 'badge ok';
      } catch (e) {
        badge.textContent = 'OFFLINE';
        badge.className = 'badge err';
      }
    }

    async function loadProjects() {
      const sel = document.getElementById('projectSelect');
      sel.innerHTML = '<option>Se încarcă...</option>';
      try {
        const res = await fetch('/api/projects', { headers: authHeaders() });
        if (!res.ok) {
          sel.innerHTML = `<option value="">HTTP ${res.status} (verifică CORTEX_API_KEY)</option>`;
          return;
        }
        const list = await res.json();
        if (!Array.isArray(list) || list.length === 0) {
          sel.innerHTML = '<option value="">Niciun proiect în DB</option>';
          return;
        }
        sel.innerHTML = list.map(p =>
          `<option value="${p.id}">#${p.id} — ${p.name} [${p.project_type}]</option>`
        ).join('');
      } catch (e) {
        sel.innerHTML = `<option value="">Eroare: ${e.message}</option>`;
      }
    }

    async function loadOllamaModels() {
      const sel = document.getElementById('ollamaModelSelect');
      sel.innerHTML = '<option value="">Se încarcă modelele...</option>';
      try {
        const res = await fetch('/api/ollama/models', { headers: authHeaders() });
        if (!res.ok) {
          const txt = await res.text();
          sel.innerHTML = `<option value="">Eroare HTTP ${res.status}: ${txt}</option>`;
          return;
        }
        const data = await res.json();
        const models = Array.isArray(data.models) ? data.models : [];
        if (models.length === 0) {
          sel.innerHTML = '<option value="">Niciun model în Ollama</option>';
          return;
        }
        sel.innerHTML = models.map(m => {
          const extra = m.parameter_size ? ` (${m.parameter_size})` : '';
          return `<option value="${m.name}">${m.name}${extra}</option>`;
        }).join('');
      } catch (e) {
        sel.innerHTML = `<option value="">Eroare: ${e.message}</option>`;
      }
    }

    function loadProjectsAndModels() {
      loadProjects();
      if (document.getElementById('providerSelect').value === 'ollama') {
        loadOllamaModels();
      }
    }

    async function sendChat() {
      const projectId = Number(document.getElementById('projectSelect').value);
      const llm_provider = document.getElementById('providerSelect').value;
      const ollamaModel = document.getElementById('ollamaModelSelect').value || null;
      const message = document.getElementById('chatPrompt').value.trim();
      const activePath = document.getElementById('activePath').value.trim();
      const activeContent = document.getElementById('activeContent').value;
      const out = document.getElementById('chatOutput');
      const meta = document.getElementById('usageMeta');
      const btn = document.getElementById('sendBtn');

      if (!projectId) { out.textContent = 'Selectează un proiect valid.'; return; }
      if (!message) { out.textContent = 'Introdu o întrebare.'; return; }

      btn.disabled = true;
      out.textContent = 'Se caută în pgvector și se generează răspunsul...';
      meta.textContent = '';

      const payload = {
        project_id: projectId,
        message,
        llm_provider,
        ollama_model: llm_provider === 'ollama' ? ollamaModel : null,
        active_file: activePath ? { relative_path: activePath, content: activeContent } : null,
        mentioned_files: []
      };

      try {
        const t0 = performance.now();
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: authHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify(payload)
        });
        const text = await res.text();
        const dt = ((performance.now() - t0) / 1000).toFixed(2);
        if (!res.ok) {
          out.textContent = `HTTP ${res.status}: ${text}`;
          return;
        }
        const data = JSON.parse(text);
        out.textContent = data.answer || '(răspuns gol)';
        const u = data.usage || {};
        meta.textContent = u.total_tokens
          ? `${dt}s | tokeni: prompt=${u.prompt_tokens ?? 0}, răspuns=${u.candidates_tokens ?? 0}, total=${u.total_tokens}`
          : `${dt}s`;
      } catch (e) {
        out.textContent = 'Eroare rețea: ' + e.message;
      } finally {
        btn.disabled = false;
      }
    }

    async function sendIndex() {
      const projectId = Number(document.getElementById('projectSelect').value);
      const relative_path = document.getElementById('idxPath').value.trim();
      const content = document.getElementById('idxContent').value;
      const out = document.getElementById('idxOutput');
      const btn = document.getElementById('idxBtn');

      if (!projectId) { out.textContent = 'Selectează un proiect valid.'; return; }

      btn.disabled = true;
      out.textContent = 'Se calculează SHA-256 / embeddings...';

      try {
        const res = await fetch('/api/index', {
          method: 'POST',
          headers: authHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({
            project_id: projectId,
            files: [{ relative_path, content }]
          })
        });
        const text = await res.text();
        out.textContent = res.ok ? JSON.stringify(JSON.parse(text), null, 2) : `HTTP ${res.status}: ${text}`;
      } catch (e) {
        out.textContent = 'Eroare rețea: ' + e.message;
      } finally {
        btn.disabled = false;
      }
    }

    checkHealth();
    loadProjects();
  </script>
</body>
</html>"#;