use crate::config::{gemini_api_key, gemini_chat_model, ollama_chat_model, ollama_url};
use crate::indexer::CodeIndexer;
use crate::types::{RemoteFilePayload, TokenUsageDto};
use cortex_parser::parser_for_project_type;
use pgvector::Vector;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Row};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub type BoxError = Box<dyn Error + Send + Sync>;

pub struct ApiChatReply {
    pub answer: String,
    pub usage: TokenUsageDto,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LlmProvider {
    Ollama,
    Google,
}

impl LlmProvider {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "ollama" | "local" => Some(Self::Ollama),
            "google" | "gemini" => Some(Self::Google),
            _ => None,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Ollama => "ollama",
            Self::Google => "google",
        }
    }
}

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
struct OllamaOptions {
    num_ctx: u32,
}

#[derive(Serialize)]
struct ChatReq {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Deserialize)]
struct ChatResMessage {
    content: String,
}

#[derive(Deserialize)]
struct ChatRes {
    message: ChatResMessage,
}

#[derive(Serialize)]
struct GeminiPart {
    text: String,
}

#[derive(Serialize)]
struct GeminiContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    parts: Vec<GeminiPart>,
}

#[derive(Serialize)]
struct GeminiGenerateReq {
    #[serde(rename = "systemInstruction", skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiContent>,
    contents: Vec<GeminiContent>,
}

#[derive(Deserialize)]
struct GeminiCandidateContent {
    parts: Option<Vec<GeminiPartResp>>,
}

#[derive(Deserialize)]
struct GeminiPartResp {
    text: Option<String>,
}

#[derive(Deserialize)]
struct GeminiCandidate {
    content: Option<GeminiCandidateContent>,
}

#[derive(Deserialize)]
struct GeminiApiError {
    message: String,
}

#[derive(Deserialize)]
struct GeminiUsageMetadata {
    #[serde(rename = "promptTokenCount")]
    prompt_token_count: Option<u32>,
    #[serde(rename = "candidatesTokenCount")]
    candidates_token_count: Option<u32>,
    #[serde(rename = "totalTokenCount")]
    total_token_count: Option<u32>,
}

#[derive(Deserialize)]
struct GeminiGenerateRes {
    candidates: Option<Vec<GeminiCandidate>>,
    error: Option<GeminiApiError>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<GeminiUsageMetadata>,
}

const MAX_HISTORY_MESSAGES: usize = 20;
const MAX_FULL_FILE_BYTES: usize = 65_000;
const OLLAMA_NUM_CTX: u32 = 32_768;

pub struct RagChat {
    pool: PgPool,
    client: reqwest::Client,
}

impl RagChat {
    pub fn new(pool: PgPool) -> Self {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(90))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self { pool, client }
    }

    async fn get_project_name(&self, project_id: i64) -> String {
        sqlx::query_scalar::<_, String>("SELECT name FROM brain.projects WHERE id = $1")
            .bind(project_id)
            .fetch_optional(&self.pool)
            .await
            .ok()
            .flatten()
            .unwrap_or_else(|| format!("#{}", project_id))
    }


    fn file_identity_keys(path_str: &str, disk_path: Option<&Path>) -> Vec<String> {
        let mut keys = Vec::new();
        let norm = path_str.replace('\\', "/").to_ascii_lowercase();
        if !norm.is_empty() {
            keys.push(norm.clone());
        }
        if let Some(p) = disk_path {
            if let Ok(canon) = std::fs::canonicalize(p) {
                keys.push(canon.to_string_lossy().replace('\\', "/").to_ascii_lowercase());
            }
        }
        if let Some(fname) = Path::new(&norm).file_name().and_then(|n| n.to_str()) {
            if !fname.is_empty() {
                keys.push(format!("fname:{}", fname));
            }
        }
        keys
    }

    fn already_seen_or_mark(seen: &mut HashSet<String>, keys: &[String]) -> bool {
        if keys.iter().any(|k| seen.contains(k)) {
            return true;
        }
        for k in keys {
            seen.insert(k.clone());
        }
        false
    }

    fn extract_at_targets(text: &str) -> HashSet<String> {
        text.split_whitespace()
            .filter_map(|raw| {
                let trimmed = raw.trim_matches(|c: char| {
                    matches!(
                        c,
                        ',' | ';'
                            | ':'
                            | '!'
                            | '?'
                            | '('
                            | ')'
                            | '['
                            | ']'
                            | '{'
                            | '}'
                            | '"'
                            | '\''
                    )
                });
                trimmed
                    .strip_prefix('@')
                    .map(|t| t.trim().replace('\\', "/").to_ascii_lowercase())
            })
            .filter(|t| !t.is_empty())
            .collect()
    }

    fn read_from_project_disk(root_path: Option<&str>, rel_path: &str) -> Option<String> {
        let candidate = if Path::new(rel_path).is_absolute() {
            PathBuf::from(rel_path)
        } else {
            let root = Path::new(root_path?.trim());
            if !root.is_dir() {
                return None;
            }
            let clean_rel = rel_path.replace('\\', "/");
            let mut safe_rel = PathBuf::new();
            for comp in Path::new(&clean_rel).components() {
                match comp {
                    std::path::Component::Normal(p) => safe_rel.push(p),
                    std::path::Component::CurDir => {}
                    _ => return None,
                }
            }
            if safe_rel.as_os_str().is_empty() {
                return None;
            }
            let joined = root.join(&safe_rel);
            if let (Ok(canon_root), Ok(canon_file)) = (root.canonicalize(), joined.canonicalize()) {
                if !canon_file.starts_with(&canon_root) {
                    return None;
                }
            }
            joined
        };

        if !candidate.is_file() {
            return None;
        }
        std::fs::read_to_string(&candidate).ok()
    }

    /// Caută fișierele menționate (ex: "@vi.c") live pe disc din root_path
    async fn fetch_mentioned_files(
        &self,
        project_id: i64,
        query: &str,
    ) -> Result<Vec<(String, String)>, BoxError> {
        let targets = Self::extract_at_targets(query);
        if targets.is_empty() {
            return Ok(Vec::new());
        }

        let root_path: Option<String> =
            sqlx::query_scalar("SELECT root_path FROM brain.projects WHERE id = $1")
                .bind(project_id)
                .fetch_optional(&self.pool)
                .await?
                .flatten();

        let rows = sqlx::query("SELECT path FROM brain.source_files WHERE project_id = $1")
            .bind(project_id)
            .fetch_all(&self.pool)
            .await?;

        let mut full_files = Vec::new();
        let mut seen_keys = HashSet::new();

        for row in rows {
            let path: String = row.get("path");
            let path_lower = path.replace('\\', "/").to_ascii_lowercase();
            let file_name = Path::new(&path_lower)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(&path_lower);

            let mentioned = targets.iter().any(|target| {
                target.as_str() == file_name
                    || target.as_str() == path_lower
                    || path_lower.ends_with(&format!("/{}", target))
            });

            if !mentioned {
                continue;
            }

            let disk_candidate = if Path::new(&path).is_absolute() {
                Some(PathBuf::from(&path))
            } else {
                root_path.as_ref().map(|root| Path::new(root).join(&path))
            };

            let keys = Self::file_identity_keys(&path, disk_candidate.as_deref());
            if Self::already_seen_or_mark(&mut seen_keys, &keys) {
                continue;
            }

            if let Some(content) = Self::read_from_project_disk(root_path.as_deref(), &path) {
                if content.len() <= MAX_FULL_FILE_BYTES {
                    println!(
                        "   📄 Atașez fișierul complet (HDD): {} ({} bytes)",
                        path,
                        content.len()
                    );
                    let formatted = format!("=== FIȘIER COMPLET: {} ===\n{}", path, content);
                    full_files.push((path, formatted));
                } else {
                    println!(
                        "   ⚠️ Fișierul {} e prea mare pentru atașare integrală ({} bytes), folosesc chunk-urile RAG.",
                        path,
                        content.len()
                    );
                }
            }
        }

        for target in &targets {
            if let Some(content) = Self::read_from_project_disk(root_path.as_deref(), target) {
                let disk_candidate = root_path
                    .as_ref()
                    .map(|root| Path::new(root).join(target));
                let keys = Self::file_identity_keys(target, disk_candidate.as_deref());
                if Self::already_seen_or_mark(&mut seen_keys, &keys) {
                    continue;
                }
                if content.len() <= MAX_FULL_FILE_BYTES {
                    println!(
                        "   📄 Atașez fișier nou de pe disc: {} ({} bytes)",
                        target,
                        content.len()
                    );
                    let formatted = format!("=== FIȘIER COMPLET: {} ===\n{}", target, content);
                    full_files.push((target.clone(), formatted));
                }
            }
        }

        Ok(full_files)
    }

    /// Atașează direct de pe HDD (din `root_path`) primele `max_files` fișiere găsite în `context_list`
    fn attach_top_rag_files_from_disk(
        root_path: Option<&str>,
        context_list: &[String],
        full_files: &mut Vec<(String, String)>,
        seen_keys: &mut HashSet<String>,
        max_files: usize,
    ) {
        let mut attached = 0usize;
        for ctx in context_list {
            if attached >= max_files {
                break;
            }
            let rel_path = ctx
                .lines()
                .next()
                .and_then(|l| l.split_once("Fișier: "))
                .map(|(_, rest)| rest.split(" [chunk").next().unwrap_or(rest))
                .map(|s| s.split(" | ").next().unwrap_or(s).trim())
                .unwrap_or("");

            if rel_path.is_empty() {
                continue;
            }

            let keys = Self::file_identity_keys(rel_path, None);
            if seen_keys.iter().any(|k| keys.contains(k)) {
                continue;
            }

            if let Some(disk_content) = Self::read_from_project_disk(root_path, rel_path) {
                if !disk_content.is_empty() && disk_content.len() <= MAX_FULL_FILE_BYTES {
                    Self::already_seen_or_mark(seen_keys, &keys);
                    println!(
                        "   📂 Atașez din RAG direct de pe HDD: {} ({} bytes)",
                        rel_path,
                        disk_content.len()
                    );
                    let formatted =
                        format!("=== FIȘIER COMPLET (HDD): {} ===\n{}", rel_path, disk_content);
                    full_files.push((rel_path.to_string(), formatted));
                    attached += 1;
                }
            }
        }
    }

    async fn call_gemini_with_usage(
        &self,
        system_instruction: Option<&str>,
        contents: Vec<GeminiContent>,
    ) -> Result<(String, TokenUsageDto), BoxError> {
        let api_key =
            gemini_api_key().map_err(|_| "Lipsește GEMINI_API_KEY din fișierul .env!")?;
        let model = gemini_chat_model();
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
            model
        );

        let sys = system_instruction.map(|text| GeminiContent {
            role: None,
            parts: vec![GeminiPart {
                text: text.to_string(),
            }],
        });

        let total_chars: usize = contents
            .iter()
            .flat_map(|c| c.parts.iter())
            .map(|p| p.text.chars().count())
            .sum();
        let sys_chars = sys
            .as_ref()
            .map(|s| s.parts.iter().map(|p| p.text.chars().count()).sum::<usize>())
            .unwrap_or(0);

        eprintln!(
            "   ✨ [GEMINI] Trimis la {} | model: {} | contents: {} | caractere: {} (sys: {})",
            url,
            model,
            contents.len(),
            total_chars + sys_chars,
            sys_chars
        );

        let log_full_prompt = std::env::var("CORTEX_LOG_PROMPT")
            .map(|v| {
                matches!(
                    v.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "full"
                )
            })
            .unwrap_or(false);

        if log_full_prompt {
            eprintln!("   ================ PAYLOAD GEMINI START ================");
            if let Some(s) = &sys {
                let sys_text: String = s.parts.iter().map(|p| p.text.as_str()).collect();
                eprintln!("   --- [system_instruction] ---\n{}", sys_text);
            }
            for (idx, c) in contents.iter().enumerate() {
                let role = c.role.as_deref().unwrap_or("user");
                let text: String = c.parts.iter().map(|p| p.text.as_str()).collect();
                eprintln!("   --- [{}] role={} ---\n{}", idx, role, text);
            }
            eprintln!("   ================= PAYLOAD GEMINI END =================");
        }

        let req = GeminiGenerateReq {
            system_instruction: sys,
            contents,
        };

        let response = self
            .client
            .post(&url)
            .header("x-goog-api-key", api_key)
            .json(&req)
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            if let Ok(parsed) = serde_json::from_str::<GeminiGenerateRes>(&body) {
                if let Some(err) = parsed.error {
                    return Err(format!("Gemini HTTP {}: {}", status, err.message).into());
                }
            }
            return Err(format!("Gemini HTTP {}: {}", status, body).into());
        }

        let res: GeminiGenerateRes = serde_json::from_str(&body)
            .map_err(|e| format!("Răspuns JSON invalid de la Gemini: {} | Body: {}", e, body))?;

        if let Some(err) = res.error {
            return Err(format!("Eroare Gemini API: {}", err.message).into());
        }

        let usage = if let Some(u) = &res.usage_metadata {
            println!(
                "   📊 Tokeni Gemini: prompt = {}, răspuns = {}, total = {}",
                u.prompt_token_count.unwrap_or(0),
                u.candidates_token_count.unwrap_or(0),
                u.total_token_count.unwrap_or(0),
            );
            TokenUsageDto {
                prompt_tokens: u.prompt_token_count,
                candidates_tokens: u.candidates_token_count,
                total_tokens: u.total_token_count,
            }
        } else {
            TokenUsageDto::default()
        };

        let answer = res
            .candidates
            .and_then(|c| c.into_iter().next())
            .and_then(|c| c.content)
            .and_then(|c| c.parts)
            .map(|parts| {
                parts
                    .into_iter()
                    .filter_map(|p| p.text)
                    .collect::<Vec<_>>()
                    .join("")
            })
            .unwrap_or_default();

        if answer.is_empty() {
            return Err("Gemini a returnat un răspuns gol.".into());
        }

        Ok((answer, usage))
    }

    async fn call_gemini(
        &self,
        system_instruction: Option<&str>,
        contents: Vec<GeminiContent>,
    ) -> Result<String, BoxError> {
        let (answer, _) = self
            .call_gemini_with_usage(system_instruction, contents)
            .await?;
        Ok(answer)
    }

    async fn call_ollama(
        &self,
        system_instruction: Option<&str>,
        history: &[ChatMessage],
        prompt: &str,
        ollama_model: Option<&str>,
    ) -> Result<String, BoxError> {
        let mut messages = Vec::new();
        if let Some(sys) = system_instruction {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: sys.to_string(),
            });
        }
        messages.extend(history.iter().cloned());
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        });

        let model = ollama_model
            .map(str::trim)
            .filter(|m| !m.is_empty())
            .map(str::to_string)
            .unwrap_or_else(ollama_chat_model);

        let url = ollama_url("/api/chat");

        let total_chars: usize = messages.iter().map(|m| m.content.chars().count()).sum();
        eprintln!(
            "   🤖 [OLLAMA] Trimis la {} | model efectiv: {} | mesaje: {} | total_caractere: {}",
            url,
            model,
            messages.len(),
            total_chars
        );

        let log_full_prompt = std::env::var("CORTEX_LOG_PROMPT")
            .map(|v| {
                matches!(
                    v.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "full"
                )
            })
            .unwrap_or(false);

        if log_full_prompt {
            eprintln!("   ================ PAYLOAD OLLAMA START ================");
            for (idx, msg) in messages.iter().enumerate() {
                eprintln!("   --- [{}] role={} ---\n{}", idx, msg.role, msg.content);
            }
            eprintln!("   ================= PAYLOAD OLLAMA END =================");
        }

        let req = ChatReq {
            model,
            messages,
            stream: false,
            options: OllamaOptions {
                num_ctx: OLLAMA_NUM_CTX,
            },
        };

        let response = self.client.post(&url).json(&req).send().await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(format!("Ollama HTTP {}: {}", status, body).into());
        }

        let res = response.json::<ChatRes>().await?;
        Ok(res.message.content)
    }


    fn has_ambiguous_reference(query_lower: &str) -> bool {
        let tokens: Vec<String> = query_lower
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|t| !t.is_empty())
            .map(|t| t.to_string())
            .collect();

        let ambiguous_words = [
            "o", "îl", "il", "ea", "ei", "ele", "asta", "ăsta", "ăla", "acea", "acel",
            "aceasta", "acesta", "acela", "acolo", "aici", "anterior", "anterioară",
            "precedent", "precedentă", "mai", "sus", "jos", "înainte",
        ];

        if tokens.iter().any(|t| ambiguous_words.contains(&t.as_str())) {
            return true;
        }

        // Expresii scurte tipice de continuare fără subiect explicit
        let vague_phrases = [
            "cum funcționează",
            "cum functioneaza",
            "ce face",
            "de ce",
            "arată-mi",
            "arata-mi",
            "modifică",
            "modifica",
            "rescrie",
            "testează",
            "testeaza",
            "rulează",
            "ruleaza",
        ];

        tokens.len() <= 5 && vague_phrases.iter().any(|p| query_lower.contains(p))
    }

    fn looks_like_console_or_code(query: &str) -> bool {
        let line_count = query.lines().count();
        if line_count > 2 || query.chars().count() > 180 {
            return true;
        }

        let markers = [
            "root@",
            "~/",
            "[asm]",
            "[pasul ",
            "[debug]",
            "error[",
            "error:",
            "warning:",
            "--- [",
            "```",
        ];
        let lower = query.to_ascii_lowercase();
        markers.iter().any(|m| lower.contains(m))
    }

    fn should_skip_contextualize(query: &str, history: &[ChatMessage]) -> bool {
        if history.is_empty() {
            return true;
        }
        let trimmed = query.trim();
        if trimmed.contains('@') || Self::looks_like_console_or_code(trimmed) {
            return true;
        }

        let lower = trimmed.to_ascii_lowercase();
        if Self::has_ambiguous_reference(&lower) {
            return false;
        }

        // Dacă e o întrebare clară (>= 4 cuvinte) fără pronume/referințe ambigue, nu o rescriem
        let word_count = lower.split_whitespace().count();
        word_count >= 4
    }




    async fn contextualize_query(
        &self,
        query: &str,
        history: &[ChatMessage],
        provider: LlmProvider,
        ollama_model: Option<&str>,
    ) -> Result<String, BoxError> {
        let trimmed_query = query.trim();
        if trimmed_query.is_empty() || Self::should_skip_contextualize(trimmed_query, history) {
            return Ok(trimmed_query.to_string());
        }

        // Păstrăm doar ultimele 2 schimburi (max 4 mesaje), trunchiate, ca să nu polueze contextul
        let recent: Vec<String> = history
            .iter()
            .rev()
            .take(4)
            .rev()
            .map(|m| {
                let single_line: String = m
                    .content
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ");
                let clipped: String = single_line.chars().take(240).collect();
                format!("{}: {}", m.role, clipped)
            })
            .collect();

        let prompt = format!(
            "Istoric recent:\n{}\n\nÎntrebare nouă: {}\n\nReguli:\n1. Rescrie întrebarea nouă DOAR dacă folosește pronume sau referințe ambigue la istoricul recent, înlocuind referința ambiguă cu numele exact al funcției/fișierului/subiectului din istoric.\n2. Dacă întrebarea este deja clară sau nu are referință clară în istoric, returnează EXACT întrebarea nouă.\n3. Răspunde pe o SINGURĂ linie, fără output de consolă, fără ghilimele și fără explicații.",
            recent.join("\n"),
            trimmed_query
        );

        let raw = match provider {
            LlmProvider::Ollama => self.call_ollama(None, &[], &prompt, ollama_model).await?,
            LlmProvider::Google => {
                let contents = vec![GeminiContent {
                    role: Some("user".to_string()),
                    parts: vec![GeminiPart { text: prompt }],
                }];
                self.call_gemini(None, contents).await?
            }
        };

        // Luăm doar prima linie non-goală și limităm lungimea pentru embedding
        let first_line = raw
            .lines()
            .map(str::trim)
            .find(|l| !l.is_empty())
            .unwrap_or(trimmed_query);

        let cleaned = first_line
            .trim_matches(|c| matches!(c, '"' | '\'' | '`'))
            .trim();

        let final_query: String = if cleaned.is_empty() || cleaned.chars().count() > 220 {
            trimmed_query.chars().take(220).collect()
        } else {
            cleaned.to_string()
        };

        if final_query != trimmed_query {
            println!("   🔎 Caut după: {}", final_query);
        }
        Ok(final_query)
    }

    async fn get_query_embedding(&self, query: &str) -> Result<Vec<f32>, BoxError> {
        let req = EmbedReq {
            model: "nomic-embed-text".to_string(),
            prompt: query.to_string(),
        };

        let response = self
            .client
            .post(ollama_url("/api/embeddings"))
            .json(&req)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(format!("Ollama embeddings HTTP {}: {}", status, body).into());
        }

        let res = response.json::<EmbedRes>().await?;
        Ok(res.embedding)
    }

    async fn retrieve_context(
        &self,
        project_id: i64,
        query_vector: Vec<f32>,
    ) -> Result<Vec<String>, BoxError> {
        let vector = Vector::from(query_vector);

        let rows = sqlx::query(
            r#"
            SELECT DISTINCT ON (content)
                content,
                content_type,
                (1 - (embedding_768 <=> $1)) AS similarity
            FROM brain.memories
            WHERE project_id = $2
              AND embedding_768 IS NOT NULL
            ORDER BY content, embedding_768 <=> $1
            "#,
        )
        .bind(vector)
        .bind(project_id)
        .fetch_all(&self.pool)
        .await?;

        let mut code_matches: Vec<(f64, String, String)> = Vec::new();
        let mut func_matches: Vec<(f64, String, String)> = Vec::new();
        let mut doc_matches: Vec<(f64, String, String)> = Vec::new();

        for row in rows {
            let similarity: f64 = row.get("similarity");
            if similarity < 0.48 {
                continue;
            }
            let content_type: String = row.get("content_type");
            let content: String = row.get("content");

            match content_type.as_str() {
                "documentation" => doc_matches.push((similarity, content_type, content)),
                "function" => func_matches.push((similarity, content_type, content)),
                "source_code" => code_matches.push((similarity, content_type, content)),
                _ => {}
            }
        }

        code_matches.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        func_matches.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        doc_matches.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

        let selected_docs: Vec<_> = doc_matches.into_iter().take(1).collect();
        let selected_funcs: Vec<_> = func_matches.into_iter().take(2).collect();
        let code_quota = 8 - selected_docs.len() - selected_funcs.len();

        let mut per_file_count: HashMap<String, usize> = HashMap::new();
        let mut selected_code = Vec::with_capacity(code_quota);
        let mut overflow_code = Vec::new();

        for item in code_matches {
            if selected_code.len() >= code_quota {
                break;
            }
            let file_key = item
                .2
                .lines()
                .next()
                .and_then(|l| l.strip_prefix("Fișier: "))
                .and_then(|rest| rest.split(" [chunk").next())
                .unwrap_or("")
                .to_string();

            let count = per_file_count.entry(file_key).or_insert(0);
            if *count < 2 {
                *count += 1;
                selected_code.push(item);
            } else {
                overflow_code.push(item);
            }
        }

        for item in overflow_code {
            if selected_code.len() >= code_quota {
                break;
            }
            selected_code.push(item);
        }

        let mut combined =
            Vec::with_capacity(selected_code.len() + selected_funcs.len() + selected_docs.len());
        combined.extend(selected_code);
        combined.extend(selected_funcs);
        combined.extend(selected_docs);
        combined.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

        let mut contexts = Vec::with_capacity(combined.len());
        for (similarity, content_type, content) in combined {
            println!(
                "   🔍 Found match [{:.2}%] ({}): {}...",
                similarity * 100.0,
                content_type,
                content.lines().next().unwrap_or("")
            );
            contexts.push(format!("[{}] {}", content_type, content));
        }

        Ok(contexts)
    }

    async fn generate_answer_with_usage(
        &self,
        prompt: &str,
        history: &[ChatMessage],
        provider: LlmProvider,
        ollama_model: Option<&str>,
    ) -> Result<(String, TokenUsageDto), BoxError> {
        let system_instruction = "Ești un inginer senior de software. Răspunde ÎNTOTDEAUNA în limba română, concis, direct și tehnic. Folosește EXCLUSIV tipurile, variabilele, convențiile și funcțiile care apar explicit în CONTEXT DIN PROIECT. Este STRICT INTERZIS să inventezi funcții, structuri sau fișiere care nu există în context și să repeți același sfat generic la mai multe puncte. Dacă o bucată de cod lipsește din context, menționează exact ce lipsește.";

        match provider {
            LlmProvider::Ollama => {
                let answer = self
                    .call_ollama(Some(system_instruction), history, prompt, ollama_model)
                    .await?;
                Ok((answer, TokenUsageDto::default()))
            }
            LlmProvider::Google => {
                let mut contents: Vec<GeminiContent> = history
                    .iter()
                    .map(|msg| GeminiContent {
                        role: Some(if msg.role == "assistant" {
                            "model".to_string()
                        } else {
                            "user".to_string()
                        }),
                        parts: vec![GeminiPart {
                            text: msg.content.clone(),
                        }],
                    })
                    .collect();

                contents.push(GeminiContent {
                    role: Some("user".to_string()),
                    parts: vec![GeminiPart {
                        text: prompt.to_string(),
                    }],
                });

                self.call_gemini_with_usage(Some(system_instruction), contents)
                    .await
            }
        }
    }

    async fn generate_answer(
        &self,
        prompt: &str,
        history: &[ChatMessage],
        provider: LlmProvider,
        ollama_model: Option<&str>,
    ) -> Result<String, BoxError> {
        let (answer, _) = self
            .generate_answer_with_usage(prompt, history, provider, ollama_model)
            .await?;
        Ok(answer)
    }

    async fn ensure_chat_history_table(&self) -> Result<(), BoxError> {
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

    async fn load_chat_history(&self, project_id: i64) -> Result<Vec<ChatMessage>, BoxError> {
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
    ) -> Result<(), BoxError> {
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

    /// Endpoint-ul apelat de `server.rs` (`POST /api/chat`) pentru clienții externi (ex. RED)
    pub async fn ask_from_api(
        &self,
        project_id: i64,
        query: &str,
        provider: &str,
        ollama_model: Option<String>,
        active_file: Option<RemoteFilePayload>,
        mentioned_files: Vec<RemoteFilePayload>,
    ) -> Result<ApiChatReply, BoxError> {
        self.ensure_chat_history_table().await?;
        let conversation_history = self.load_chat_history(project_id).await?;

        let active_provider = match LlmProvider::from_str(provider) {
            Some(LlmProvider::Google) if gemini_api_key().is_ok() => LlmProvider::Google,
            Some(LlmProvider::Ollama) => LlmProvider::Ollama,
            _ if gemini_api_key().is_ok() => LlmProvider::Google,
            _ => LlmProvider::Ollama,
        };

        let search_query = self
            .contextualize_query(
                query,
                &conversation_history,
                active_provider,
                ollama_model.as_deref(),
            )
            .await
            .unwrap_or_else(|_| query.to_string());

        let root_path: Option<String> =
            sqlx::query_scalar("SELECT root_path FROM brain.projects WHERE id = $1")
                .bind(project_id)
                .fetch_optional(&self.pool)
                .await
                .ok()
                .flatten()
                .flatten();

        let mut full_files: Vec<(String, String)> = Vec::new();
        let mut seen_keys: HashSet<String> = HashSet::new();

        for file in active_file.into_iter().chain(mentioned_files.into_iter()) {
            let content =
                Self::read_from_project_disk(root_path.as_deref(), &file.relative_path)
                    .unwrap_or(file.content);

            if content.is_empty() || content.len() > MAX_FULL_FILE_BYTES {
                continue;
            }
            let keys = Self::file_identity_keys(&file.relative_path, None);
            if Self::already_seen_or_mark(&mut seen_keys, &keys) {
                continue;
            }
            let formatted = format!(
                "=== FIȘIER COMPLET: {} ===\n{}",
                file.relative_path, content
            );
            full_files.push((file.relative_path, formatted));
        }

        if let Ok(local_files) = self.fetch_mentioned_files(project_id, query).await {
            for (path, formatted) in local_files {
                let keys = Self::file_identity_keys(&path, None);
                if Self::already_seen_or_mark(&mut seen_keys, &keys) {
                    continue;
                }
                full_files.push((path, formatted));
            }
        }

        let query_vector = self.get_query_embedding(&search_query).await?;
        let context_list = self.retrieve_context(project_id, query_vector).await?;

        Self::attach_top_rag_files_from_disk(
            root_path.as_deref(),
            &context_list,
            &mut full_files,
            &mut seen_keys,
            2,
        );

        let filtered_chunks: Vec<String> = context_list
            .into_iter()
            .filter(|ctx| {
                let ctx_norm = ctx.replace('\\', "/").to_ascii_lowercase();
                !full_files.iter().any(|(path, _)| {
                    if path.is_empty() {
                        return false;
                    }
                    let path_norm = path.replace('\\', "/").to_ascii_lowercase();
                    let fname = Path::new(&path_norm)
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or(&path_norm);
                    ctx_norm.contains(&path_norm) || ctx_norm.contains(fname)
                })
            })
            .collect();

        let mut context_blocks: Vec<String> = full_files
            .into_iter()
            .map(|(_, formatted)| formatted)
            .collect();
        context_blocks.extend(filtered_chunks);
        let context_str = context_blocks.join("\n---\n");

        let full_prompt = format!(
            "CONTEXT DIN PROIECT:\n{}\n\nÎNTREBARE UTILIZATOR: {}\n\nFurnizează un răspuns clar și direct bazat pe contextul de mai sus.",
            context_str, query
        );

        let (answer, usage) = self
            .generate_answer_with_usage(
                &full_prompt,
                &conversation_history,
                active_provider,
                ollama_model.as_deref(),
            )
            .await?;

        self.save_chat_exchange(project_id, query, &answer).await?;

        Ok(ApiChatReply { answer, usage })
    }

    async fn handle_command(
        &self,
        input: &str,
        active_project_id: &mut i64,
        active_provider: &mut LlmProvider,
    ) -> Result<bool, BoxError> {
        let words: Vec<&str> = input.split_whitespace().collect();
        let Some(command) = words.first().map(|word| word.to_ascii_lowercase()) else {
            return Ok(false);
        };

        match command.as_str() {
            "help" => {
                println!("Comenzi disponibile:");
                println!("  use llm <ollama|google>            - comută între Ollama local și Google Gemini");
                println!("  llm                                - afișează providerul LLM activ");
                println!("  use project <nume>                 - selectează proiectul pentru chat (alias: set project <nume>)");
                println!("  projects                           - listează proiectele");
                println!("  add project <nume> <cale> [tip]    - adaugă și indexează un proiect (tip: cpp_code, c_code, oli_language, book)");
                println!("  reindex project <nume>             - reindexează proiectul");
                println!("  delete project <nume>              - șterge proiectul și datele indexate");
                println!("  clear hist                         - șterge istoricul chatului pentru proiectul curent");
                println!("  help                               - afișează această listă");
                println!("  exit                               - închide chatul");
                println!("  (Poți menționa orice fișier cu @nume_fisier pentru a-l atașa integral)");
                Ok(true)
            }
            "llm" if words.len() == 1 => {
                let model = match active_provider {
                    LlmProvider::Ollama => ollama_chat_model(),
                    LlmProvider::Google => gemini_chat_model(),
                };
                println!(
                    "Provider LLM activ: {} ({}) | Schimbă cu: use llm ollama | use llm google",
                    active_provider.label(),
                    model
                );
                Ok(true)
            }
            "clear" => {
                if words.len() == 2 && words[1].eq_ignore_ascii_case("hist") {
                    sqlx::query("DELETE FROM brain.chat_history WHERE project_id = $1")
                        .bind(*active_project_id)
                        .execute(&self.pool)
                        .await?;
                    println!("Istoricul conversației pentru proiectul curent a fost șters.");
                } else {
                    println!("Utilizare: clear hist");
                }
                Ok(true)
            }
            "projects" if words.len() == 1 => {
                let rows = sqlx::query(
                    "SELECT name, project_type FROM brain.projects ORDER BY name",
                )
                .fetch_all(&self.pool)
                .await?;
                if rows.is_empty() {
                    println!("Nu există proiecte.");
                } else {
                    println!("Proiecte:");
                    for row in rows {
                        let name: String = row.get("name");
                        let ptype: String = row.get("project_type");
                        println!("  - {} [{}]", name, ptype);
                    }
                }
                Ok(true)
            }
            "add" => {
                if words.len() < 4 || !words[1].eq_ignore_ascii_case("project") {
                    println!("Utilizare: add project <nume> <cale> [tip]");
                    return Ok(true);
                }

                let project_name = words[2];
                let (source_path, project_type) = if words.len() >= 5
                    && matches!(
                        words[words.len() - 1].to_ascii_lowercase().as_str(),
                        "code" | "book" | "documentation"
                    ) {
                    let path = words[3..words.len() - 1]
                        .join(" ")
                        .trim_matches('"')
                        .trim_matches('\'')
                        .to_string();
                    (path, words[words.len() - 1].to_ascii_lowercase())
                } else {
                    let path = words[3..]
                        .join(" ")
                        .trim_matches('"')
                        .trim_matches('\'')
                        .to_string();
                    (path, "code".to_string())
                };

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
                    "INSERT INTO brain.projects (name, project_type, root_path) VALUES ($1, $2, $3) RETURNING id",
                )
                .bind(project_name)
                .bind(&project_type)
                .bind(&source_path)
                .fetch_one(&self.pool)
                .await?;

                let parser = parser_for_project_type(&project_type);
                if let Err(error) = CodeIndexer::new(self.pool.clone())
                    .reindex_and_store(project_id, &source_path, parser.as_ref())
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
                    "Proiectul '{}' [{}] a fost adăugat, indexat și selectat pentru chat.",
                    project_name, project_type
                );
                Ok(true)
            }
            "use" | "set" => {
                if words.len() == 3 && words[1].eq_ignore_ascii_case("llm") {
                    match LlmProvider::from_str(words[2]) {
                        Some(LlmProvider::Google) => {
                            if gemini_api_key().is_err() {
                                println!(
                                    "⚠️ Lipsește GEMINI_API_KEY în .env — rămân pe {}.",
                                    active_provider.label()
                                );
                            } else {
                                *active_provider = LlmProvider::Google;
                                println!(
                                    "✅ Comutat pe Google Gemini ({}).",
                                    gemini_chat_model()
                                );
                            }
                        }
                        Some(LlmProvider::Ollama) => {
                            *active_provider = LlmProvider::Ollama;
                            println!("✅ Comutat pe Ollama local ({}).", ollama_chat_model());
                        }
                        None => println!("Utilizare: use llm <ollama|google>"),
                    }
                    return Ok(true);
                }

                if words.len() < 3 || !words[1].eq_ignore_ascii_case("project") {
                    println!("Utilizare: use project <nume>  sau  use llm <ollama|google>");
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
                        "SELECT id, project_type, root_path FROM brain.projects WHERE LOWER(name) = LOWER($1)",
                    )
                    .bind(&project_name)
                    .fetch_optional(&self.pool)
                    .await?;

                    let Some(project) = project else {
                        println!("Proiectul '{}' nu există.", project_name);
                        return Ok(true);
                    };
                    let project_id: i64 = project.get("id");
                    let project_type: String = project.get("project_type");
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

                    println!(
                        "Reindexez proiectul '{}' [{}] din {}...",
                        project_name, project_type, root_path
                    );
                    let parser = parser_for_project_type(&project_type);
                    CodeIndexer::new(self.pool.clone())
                        .reindex_and_store(project_id, &root_path, parser.as_ref())
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

    pub async fn start_repl(&self, mut project_id: i64) -> Result<(), BoxError> {
        let mut line_editor = DefaultEditor::new()?;
        self.ensure_chat_history_table().await?;
        let mut conversation_history = self.load_chat_history(project_id).await?;
        let mut active_project_name = self.get_project_name(project_id).await;

        let mut active_provider = match std::env::var("DEFAULT_LLM")
            .ok()
            .as_deref()
            .and_then(LlmProvider::from_str)
        {
            Some(LlmProvider::Google) if gemini_api_key().is_ok() => LlmProvider::Google,
            _ => LlmProvider::Ollama,
        };

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
            let prompt_label =
                format!("Cortex [{} | {}] > ", active_project_name, active_provider.label());
            let input = match line_editor.readline(&prompt_label) {
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

            if query.is_empty() {
                continue;
            }

            if query.eq_ignore_ascii_case("exit") {
                println!("La revedere!");
                break;
            }
            line_editor.add_history_entry(query)?;

            let previous_project_id = project_id;
            match self
                .handle_command(query, &mut project_id, &mut active_provider)
                .await
            {
                Ok(true) => {
                    if project_id != previous_project_id {
                        conversation_history = self.load_chat_history(project_id).await?;
                        active_project_name = self.get_project_name(project_id).await;
                    } else if is_clear_hist_command(query) {
                        conversation_history.clear();
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
            if !conversation_history.is_empty() {
                println!("   ⏳ Contextualizez întrebarea ({})...", active_provider.label());
            }
            let search_query = match self
                .contextualize_query(
                    query,
                    &conversation_history,
                    active_provider,
                    None,
                )
                .await
            {
                Ok(q) => q,
                Err(e) => {
                    eprintln!(
                        "   ⚠️ Eroare la contextualizare ({}): {}",
                        active_provider.label(),
                        e
                    );
                    query.to_string()
                }
            };

            let combined_mention_query = format!("{} {}", query, search_query);
            let mut full_files = match self
                .fetch_mentioned_files(project_id, &combined_mention_query)
                .await
            {
                Ok(files) => files,
                Err(e) => {
                    eprintln!("   ⚠️ Eroare la citirea fișierelor menționate: {}", e);
                    Vec::new()
                }
            };

            println!("   ⏳ Generez embedding Ollama (nomic-embed-text)...");
            let query_vector = match self.get_query_embedding(&search_query).await {
                Ok(v) => v,
                Err(e) => {
                    eprintln!(
                        "Eroare la generarea embedding-ului (verifică dacă Ollama e pornit): {}",
                        e
                    );
                    continue;
                }
            };

            println!("   ⏳ Interoghez Postgres (pgvector)...");
            let context_list = match self.retrieve_context(project_id, query_vector).await {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Eroare la extragerea contextului: {}", e);
                    continue;
                }
            };

            let root_path: Option<String> =
                sqlx::query_scalar("SELECT root_path FROM brain.projects WHERE id = $1")
                    .bind(project_id)
                    .fetch_optional(&self.pool)
                    .await
                    .ok()
                    .flatten()
                    .flatten();

            let mut seen_keys: HashSet<String> = HashSet::new();
            for (path, _) in &full_files {
                let keys = Self::file_identity_keys(path, None);
                Self::already_seen_or_mark(&mut seen_keys, &keys);
            }

            Self::attach_top_rag_files_from_disk(
                root_path.as_deref(),
                &context_list,
                &mut full_files,
                &mut seen_keys,
                2,
            );

            let filtered_chunks: Vec<String> = context_list
                .into_iter()
                .filter(|ctx| {
                    let ctx_norm = ctx.replace('\\', "/").to_ascii_lowercase();
                    !full_files.iter().any(|(path, _)| {
                        if path.is_empty() {
                            return false;
                        }
                        let path_norm = path.replace('\\', "/").to_ascii_lowercase();
                        let fname = Path::new(&path_norm)
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or(&path_norm);
                        ctx_norm.contains(&path_norm) || ctx_norm.contains(fname)
                    })
                })
                .collect();

            let mut context_blocks: Vec<String> = full_files
                .into_iter()
                .map(|(_, formatted)| formatted)
                .collect();
            context_blocks.extend(filtered_chunks);
            let context_str = context_blocks.join("\n---\n");

            let full_prompt = format!(
                "CONTEXT DIN PROIECT:\n{}\n\nÎNTREBARE UTILIZATOR: {}\n\nFurnizează un răspuns clar și direct bazat pe contextul de mai sus.",
                context_str, query
            );

            println!(
                "\n[2/2] Generez răspuns cu LLM ({})...",
                active_provider.label()
            );
            match self
                .generate_answer(&full_prompt, &conversation_history, active_provider, None)
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

fn is_clear_hist_command(input: &str) -> bool {
    let words: Vec<&str> = input.split_whitespace().collect();
    words.len() == 2 && words[0].eq_ignore_ascii_case("clear") && words[1].eq_ignore_ascii_case("hist")
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
    use super::{
        infer_root_path, is_clear_hist_command, remember_exchange, LlmProvider,
        MAX_HISTORY_MESSAGES,
    };

    #[test]
    fn parses_llm_provider_aliases() {
        assert_eq!(LlmProvider::from_str("ollama"), Some(LlmProvider::Ollama));
        assert_eq!(LlmProvider::from_str("local"), Some(LlmProvider::Ollama));
        assert_eq!(LlmProvider::from_str("google"), Some(LlmProvider::Google));
        assert_eq!(LlmProvider::from_str("gemini"), Some(LlmProvider::Google));
        assert_eq!(LlmProvider::from_str("unknown"), None);
    }

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
    fn detects_clear_hist_command() {
        assert!(is_clear_hist_command("clear hist"));
        assert!(is_clear_hist_command("  CLEAR   HIST "));
        assert!(!is_clear_hist_command("clear"));
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