use crate::config::ollama_url;
use crate::types::RemoteFilePayload;
use cortex_parser::{parser_for_project_type, CodeParser, ParsedFile};
use pgvector::Vector;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sqlx::{PgPool, Row};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub type BoxError = Box<dyn Error + Send + Sync>;

#[derive(Debug, Clone, Copy, Default)]
pub struct IndexStats {
    pub indexed_count: usize,
    pub skipped_unchanged: usize,
}

#[derive(Serialize)]
struct EmbeddingRequest {
    model: String,
    prompt: String,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
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

    /// Asigură coloanele necesare pentru indexarea incrementală SHA-256 fără a strica DB-ul existent
    async fn ensure_incremental_schema(&self) -> Result<(), BoxError> {
        sqlx::query(
            "ALTER TABLE brain.source_files ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64)",
        )
        .execute(&self.pool)
        .await?;

        let _ = sqlx::query("ALTER TABLE brain.source_files ALTER COLUMN content DROP NOT NULL")
            .execute(&self.pool)
            .await;

        sqlx::query(
            "ALTER TABLE brain.memories ADD COLUMN IF NOT EXISTS file_id BIGINT REFERENCES brain.source_files(id) ON DELETE CASCADE",
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Apel către Ollama local pentru a obține vectorul (768 dimensiuni)
    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>, BoxError> {
        let req = EmbeddingRequest {
            model: "nomic-embed-text".to_string(),
            prompt: text.to_string(),
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

        let res = response.json::<EmbeddingResponse>().await?;
        Ok(res.embedding)
    }

    /// Indexare incrementală pe bază de SHA-256: recalculează embedding-urile doar pentru fișierele modificate
    pub async fn index_and_store(
        &self,
        project_id: i64,
        dir_path: &str,
        parser: &dyn CodeParser,
    ) -> Result<(), BoxError> {
        self.ensure_incremental_schema().await?;

        println!("Scanare cod sursă în: {}", dir_path);
        let parsed_files = parser
            .parse_directory(dir_path)
            .map_err(|e| e.to_string())?;
        println!("Am găsit {} fișiere în proiect.", parsed_files.len());

        let existing_rows = sqlx::query(
            "SELECT id, path, content_hash FROM brain.source_files WHERE project_id = $1",
        )
        .bind(project_id)
        .fetch_all(&self.pool)
        .await?;

        let mut existing_map: HashMap<String, (i64, Option<String>)> = HashMap::new();
        for row in existing_rows {
            let id: i64 = row.get("id");
            let path: String = row.get("path");
            let hash: Option<String> = row.try_get("content_hash").ok().flatten();
            existing_map.insert(path, (id, hash));
        }

        let mut seen_paths = HashSet::with_capacity(parsed_files.len());
        let mut stats = IndexStats::default();

        for (idx, file) in parsed_files.iter().enumerate() {
            seen_paths.insert(file.path.clone());
            let new_hash = compute_sha256(&file.content);

            let unchanged = existing_map
                .get(&file.path)
                .and_then(|(_, old_hash)| old_hash.as_deref())
                == Some(new_hash.as_str());

            if unchanged {
                stats.skipped_unchanged += 1;
                continue;
            }

            println!(
                "\n   [{}/{}] Indexez fișier modificat/nou: {}",
                idx + 1,
                parsed_files.len(),
                file.path
            );
            self.store_file_with_hash(project_id, file, &new_hash).await?;
            stats.indexed_count += 1;
        }

        // Ștergem din DB fișierele care au fost eliminate de pe disc
        for (old_path, (old_file_id, _)) in existing_map {
            if !seen_paths.contains(&old_path) {
                println!("   🗑️  Elimin fișier șters din DB: {}", old_path);
                self.delete_file_memories(project_id, old_file_id, &old_path)
                    .await?;
                sqlx::query("DELETE FROM brain.source_files WHERE id = $1")
                    .bind(old_file_id)
                    .execute(&self.pool)
                    .await?;
            }
        }

        println!(
            "\nIndexare incrementală finalizată! ({} indexate, {} neschimbate)",
            stats.indexed_count, stats.skipped_unchanged
        );
        Ok(())
    }

    /// Șterge datele indexate ale proiectului și îl reindexează complet
    pub async fn reindex_and_store(
        &self,
        project_id: i64,
        dir_path: &str,
        parser: &dyn CodeParser,
    ) -> Result<(), BoxError> {
        sqlx::query("DELETE FROM brain.memories WHERE project_id = $1")
            .bind(project_id)
            .execute(&self.pool)
            .await?;

        sqlx::query("DELETE FROM brain.source_files WHERE project_id = $1")
            .bind(project_id)
            .execute(&self.pool)
            .await?;

        self.index_and_store(project_id, dir_path, parser).await
    }

    /// Indexează incremental fișierele trimise prin HTTP (`POST /api/index`) de către RED
    pub async fn index_remote_files(
        &self,
        project_id: i64,
        files: &[RemoteFilePayload],
        project_type: &str,
    ) -> Result<IndexStats, BoxError> {
        self.ensure_incremental_schema().await?;

        let mut stats = IndexStats::default();
        let mut changed_files: Vec<(&RemoteFilePayload, String)> = Vec::new();

        for file in files {
            let clean_rel = file.relative_path.replace('\\', "/");
            if clean_rel.is_empty() {
                continue;
            }

            let new_hash = file
                .sha256
                .clone()
                .filter(|h| !h.is_empty())
                .unwrap_or_else(|| compute_sha256(&file.content));

            let existing_hash: Option<String> = sqlx::query_scalar(
                "SELECT content_hash FROM brain.source_files WHERE project_id = $1 AND path = $2",
            )
            .bind(project_id)
            .bind(&clean_rel)
            .fetch_optional(&self.pool)
            .await?
            .flatten();

            if existing_hash.as_deref() == Some(new_hash.as_str()) {
                stats.skipped_unchanged += 1;
            } else {
                changed_files.push((file, new_hash));
            }
        }

        if changed_files.is_empty() {
            return Ok(stats);
        }

        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let temp_root =
            std::env::temp_dir().join(format!("cortex_remote_{}_{}", project_id, stamp));
        std::fs::create_dir_all(&temp_root)?;

        let mut hash_by_rel: HashMap<String, String> = HashMap::new();
        for (file, hash) in &changed_files {
            let clean_rel = file.relative_path.replace('\\', "/");
            let safe_rel = Path::new(&clean_rel)
                .components()
                .filter(|c| matches!(c, std::path::Component::Normal(_)))
                .collect::<PathBuf>();

            if safe_rel.as_os_str().is_empty() {
                continue;
            }

            let dest = temp_root.join(&safe_rel);
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&dest, &file.content)?;
            hash_by_rel.insert(
                safe_rel.to_string_lossy().replace('\\', "/"),
                hash.clone(),
            );
        }

        let temp_root_str = temp_root.to_string_lossy().into_owned();
        let parse_result = {
            let parser = parser_for_project_type(project_type);
            parser
                .parse_directory(&temp_root_str)
                .map_err(|e| e.to_string())
        };

        let store_res: Result<(), BoxError> = async {
            let mut parsed_files = parse_result?;
            for parsed in &mut parsed_files {
                let rel = Path::new(&parsed.path)
                    .strip_prefix(&temp_root)
                    .unwrap_or_else(|_| Path::new(&parsed.path))
                    .to_string_lossy()
                    .replace('\\', "/");

                if let Some(hash) = hash_by_rel.get(&rel) {
                    parsed.path = rel;
                    self.store_file_with_hash(project_id, parsed, hash).await?;
                    stats.indexed_count += 1;
                }
            }
            Ok(())
        }
        .await;

        let _ = std::fs::remove_dir_all(&temp_root);
        store_res?;
        Ok(stats)
    }

    async fn delete_file_memories(
        &self,
        project_id: i64,
        file_id: i64,
        file_path: &str,
    ) -> Result<(), BoxError> {
        let prefix_space = format!("Fișier: {} %", file_path);
        let prefix_newline = format!("Fișier: {}\n%", file_path);

        sqlx::query(
            r#"
            DELETE FROM brain.memories
            WHERE project_id = $1
              AND (file_id = $2 OR content LIKE $3 OR content LIKE $4)
            "#,
        )
        .bind(project_id)
        .bind(file_id)
        .bind(prefix_space)
        .bind(prefix_newline)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Salvează metadatele și hash-ul SHA-256 în brain.source_files (fără a duplica codul brut)
    /// și actualizează componentele, funcțiile și embedding-urile fișierului
    async fn store_file_with_hash(
        &self,
        project_id: i64,
        file: &ParsedFile,
        content_hash: &str,
    ) -> Result<(), BoxError> {
        println!("   ├─ Actualizez brain.source_files (SHA-256: {}...)...", &content_hash[..12]);
        let file_row = sqlx::query(
            r#"
            INSERT INTO brain.source_files (project_id, path, extension, content, content_hash)
            VALUES ($1, $2, $3, NULL, $4)
            ON CONFLICT (project_id, path) DO UPDATE
                SET extension = EXCLUDED.extension,
                    content = NULL,
                    content_hash = EXCLUDED.content_hash
            RETURNING id
            "#,
        )
        .bind(project_id)
        .bind(&file.path)
        .bind(&file.extension)
        .bind(content_hash)
        .fetch_one(&self.pool)
        .await?;

        let file_id: i64 = file_row.get("id");

        // Curățăm componentele, funcțiile și embedding-urile vechi ale acestui fișier
        self.delete_file_memories(project_id, file_id, &file.path)
            .await?;

        sqlx::query("DELETE FROM brain.code_components WHERE file_id = $1")
            .bind(file_id)
            .execute(&self.pool)
            .await?;
        sqlx::query("DELETE FROM brain.functions WHERE file_id = $1")
            .bind(file_id)
            .execute(&self.pool)
            .await?;

        let is_doc = matches!(file.extension.as_str(), "md" | "txt");
        let is_test = !is_doc && is_test_file(&file.path);
        let mem_type = if is_doc {
            "documentation"
        } else if is_test {
            "test_code"
        } else {
            "source_code"
        };

        // 1. Chunking pe linii cu overlap (2200 caractere, overlap 350 caractere)
        let chunks = chunk_content(&file.content, 2200, 350);
        let total_chunks = chunks.len();

        for (chunk_idx, chunk) in chunks.into_iter().enumerate() {
            let local_symbols = if is_doc {
                String::new()
            } else {
                extract_chunk_keywords(&chunk)
            };

            let chunk_text = if local_symbols.is_empty() {
                format!(
                    "Fișier: {} [chunk {}/{}]\n{}",
                    file.path,
                    chunk_idx + 1,
                    total_chunks,
                    chunk
                )
            } else {
                format!(
                    "Fișier: {} [chunk {}/{}] | Simboluri locale: {}\n{}",
                    file.path,
                    chunk_idx + 1,
                    total_chunks,
                    local_symbols,
                    chunk
                )
            };

            println!("   └─► Solicit embedding de la Ollama (nomic-embed-text)...");
            let vec = self.generate_embedding(&chunk_text).await?;
            self.insert_memory(project_id, Some(file_id), mem_type, &chunk_text, vec)
                .await?;
        }

        // 2. Salvăm componentele (clase, struct-uri, enum-uri, secțiuni Markdown)
        for comp in &file.components {
            println!(
                "   ├─ Componentă [{}]: {}",
                comp.component_type, comp.name
            );
            let comp_row = sqlx::query(
                r#"
                INSERT INTO brain.code_components (file_id, name, component_type, definition)
                VALUES ($1, $2, $3, $4)
                RETURNING id
                "#,
            )
            .bind(file_id)
            .bind(&comp.name)
            .bind(&comp.component_type)
            .bind(&comp.definition)
            .fetch_one(&self.pool)
            .await?;

            let comp_id: i64 = comp_row.get("id");

            for func in &comp.functions {
                println!("   │  ├─ Metodă: {}", func.name);
                sqlx::query(
                    r#"
                    INSERT INTO brain.functions (file_id, component_id, name, signature, description)
                    VALUES ($1, $2, $3, $4, $5)
                    "#,
                )
                .bind(file_id)
                .bind(comp_id)
                .bind(&func.name)
                .bind(&func.signature)
                .bind(&func.description)
                .execute(&self.pool)
                .await?;

                let fn_text = format!(
                    "Fișier: {} | Componentă: {} ({})\nMetodă: {}\nSemnătură: {}",
                    file.path, comp.name, comp.component_type, func.name, func.signature
                );
                let vec = self.generate_embedding(&fn_text).await?;
                let fn_mem_type = if is_test { "test_code" } else { "function" };
                self.insert_memory(project_id, Some(file_id), fn_mem_type, &fn_text, vec)
                    .await?;
            }
        }

        // 3. Salvăm funcțiile globale
        for func in &file.global_functions {
            println!("   │  ├─ Funcție globală: {}", func.name);
            sqlx::query(
                r#"
                INSERT INTO brain.functions (file_id, component_id, name, signature, description)
                VALUES ($1, NULL, $2, $3, $4)
                "#,
            )
            .bind(file_id)
            .bind(&func.name)
            .bind(&func.signature)
            .bind(&func.description)
            .execute(&self.pool)
            .await?;

            let fn_text = format!(
                "Fișier: {}\nFuncție globală: {}\nSemnătură: {}",
                file.path, func.name, func.signature
            );
            let vec = self.generate_embedding(&fn_text).await?;
            let fn_mem_type = if is_test { "test_code" } else { "function" };
            self.insert_memory(project_id, Some(file_id), fn_mem_type, &fn_text, vec)
                .await?;
        }

        Ok(())
    }

    async fn insert_memory(
        &self,
        project_id: i64,
        file_id: Option<i64>,
        content_type: &str,
        content: &str,
        embedding: Vec<f32>,
    ) -> Result<(), BoxError> {
        let vector = Vector::from(embedding);
        sqlx::query(
            r#"
            INSERT INTO brain.memories (project_id, file_id, content_type, content, embedding_768)
            VALUES ($1, $2, $3, $4, $5)
            "#,
        )
        .bind(project_id)
        .bind(file_id)
        .bind(content_type)
        .bind(content)
        .bind(vector)
        .execute(&self.pool)
        .await?;
        Ok(())
    }
}

pub fn compute_sha256(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let digest = hasher.finalize();

    let mut hex = String::with_capacity(64);
    for byte in digest {
        let _ = write!(&mut hex, "{:02x}", byte);
    }
    hex
}

/// Detectează fișierele de test după nume sau director (ex: `test.c`, `test_syntax.c`, `tests/...`)
fn is_test_file(path_str: &str) -> bool {
    let path = Path::new(path_str);
    let file_stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    if file_stem == "test"
        || file_stem.starts_with("test_")
        || file_stem.ends_with("_test")
        || file_stem.ends_with("_tests")
    {
        return true;
    }

    path.components().any(|comp| {
        comp.as_os_str()
            .to_str()
            .map(|s| matches!(s.to_ascii_lowercase().as_str(), "test" | "tests"))
            .unwrap_or(false)
    })
}

/// Extrage generic cei mai frecvenți identificatori locali dintr-un chunk (snake_case, MACRO_CASE, PascalCase, camelCase)
fn extract_chunk_keywords(chunk: &str) -> String {
    const STOPWORDS: &[&str] = &[
        "if", "else", "for", "while", "do", "switch", "case", "default", "break", "continue",
        "return", "void", "int", "char", "long", "short", "unsigned", "signed", "float", "double",
        "const", "static", "extern", "inline", "struct", "enum", "typedef", "sizeof", "include",
        "define", "ifdef", "ifndef", "endif", "class", "public", "private", "protected", "virtual",
        "override", "final", "namespace", "using", "template", "typename", "auto", "bool", "true",
        "false", "null", "nullptr", "NULL", "fn", "let", "mut", "pub", "use", "mod", "impl",
        "trait", "match", "self", "Self", "String", "Vec", "Option", "Result",
    ];

    let mut counts: HashMap<String, usize> = HashMap::new();

    for word in chunk.split(|c: char| !c.is_alphanumeric() && c != '_') {
        if word.len() < 4 || word.starts_with(|c: char| c.is_ascii_digit()) {
            continue;
        }
        if STOPWORDS.contains(&word) {
            continue;
        }

        let is_snake_or_macro = word.contains('_');
        let has_upper = word.chars().any(|c| c.is_uppercase());
        let has_lower = word.chars().any(|c| c.is_lowercase());
        let is_camel_or_pascal = has_upper && has_lower;

        if is_snake_or_macro || is_camel_or_pascal {
            *counts.entry(word.to_string()).or_insert(0) += 1;
        }
    }

    let mut ranked: Vec<(String, usize)> = counts.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    ranked
        .into_iter()
        .take(12)
        .map(|(word, _)| word)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Împarte textul în bucăți pe granițe de linie, păstrând un overlap între bucăți consecutive
fn chunk_content(content: &str, max_chars: usize, overlap_chars: usize) -> Vec<String> {
    let lines: Vec<&str> = content.lines().collect();
    let mut chunks = Vec::new();
    if lines.is_empty() {
        return chunks;
    }

    let mut start = 0;
    while start < lines.len() {
        let mut end = start;
        let mut current_len = 0;

        while end < lines.len() && current_len + lines[end].len() + 1 <= max_chars {
            current_len += lines[end].len() + 1;
            end += 1;
        }

        if end == start {
            end += 1;
        }

        chunks.push(lines[start..end].join("\n"));

        if end >= lines.len() {
            break;
        }

        let mut back_len = 0;
        let mut next_start = end;
        while next_start > start && back_len < overlap_chars {
            next_start -= 1;
            back_len += lines[next_start].len() + 1;
        }

        start = if next_start == start { end } else { next_start };
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::{chunk_content, compute_sha256, extract_chunk_keywords, is_test_file};

    #[test]
    fn computes_deterministic_sha256_hex() {
        let h1 = compute_sha256("fn main() {}");
        let h2 = compute_sha256("fn main() {}");
        let h3 = compute_sha256("fn main() { println!(\"hi\"); }");
        assert_eq!(h1.len(), 64);
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn detects_test_files_and_directories() {
        assert!(is_test_file("../nanovi/test.c"));
        assert!(is_test_file("../nanovi/test_syntax.c"));
        assert!(is_test_file("src/parser_test.rs"));
        assert!(is_test_file("project/tests/smoke.c"));
        assert!(!is_test_file("../nanovi/vi.c"));
        assert!(!is_test_file("../nanovi/nano_libc_host.c"));
    }

    #[test]
    fn chunks_on_line_boundaries_with_overlap() {
        let sample = "line_1\nline_2\nline_3\nline_4\nline_5\nline_6";
        let chunks = chunk_content(sample, 15, 7);
        assert!(chunks.len() >= 2);
        let first_last_line = chunks[0].lines().last().unwrap();
        assert!(chunks[1].contains(first_last_line));
    }

    #[test]
    fn extracts_local_identifiers_without_stopwords() {
        let snippet = "if (E.mode == MODE_COMMAND) { resolveExpression(cmd_input); cmd_input[0] = 0; }";
        let kw = extract_chunk_keywords(snippet);
        assert!(kw.contains("cmd_input"));
        assert!(kw.contains("MODE_COMMAND"));
        assert!(kw.contains("resolveExpression"));
    }
}