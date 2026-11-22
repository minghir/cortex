use regex::Regex;
use std::error::Error;
use std::fs;
use std::ops::Range;
use std::path::Path;
use walkdir::{DirEntry, WalkDir};

// --- Modele standardizate pentru orice limbaj ---
pub mod models;
pub use models::{CodeParser, ComponentInfo, FunctionInfo, ParsedFile};

/// Factory helper care alege parserul potrivit în funcție de `project_type` din DB.
/// Implicit (`"code"` sau orice tip software) detectează limbajul automat per fișier după extensie.
pub fn parser_for_project_type(project_type: &str) -> Box<dyn CodeParser> {
    match project_type.trim().to_ascii_lowercase().as_str() {
        "book" | "documentation" => Box::new(MarkdownParser),
        "c_code" => Box::new(CParser),
        "cpp_code" => Box::new(CppParser),
        "rust" | "rust_code" => Box::new(RustParser),
        _ => Box::new(PolyglotCodeParser),
    }
}

fn should_skip_entry(entry: &DirEntry) -> bool {
    if entry.depth() == 0 {
        return false;
    }
    let name = entry.file_name().to_string_lossy();
    matches!(
        name.as_ref(),
        ".git" | "target" | "node_modules" | "build" | ".idea" | ".vscode" | "__pycache__"
    )
}

// ============================================================================
// 0. Parser Poliglot (alege parserul per fișier în funcție de extensie)
// ============================================================================
pub struct PolyglotCodeParser;

impl CodeParser for PolyglotCodeParser {
    fn parse_directory(&self, dir_path: &str) -> Result<Vec<ParsedFile>, Box<dyn Error>> {
        let mut results = Vec::new();
        let re_md_heading = Regex::new(r"(?m)^(#{1,3})\s+(.+)$")?;

        for entry in WalkDir::new(dir_path)
            .into_iter()
            .filter_entry(|e| !should_skip_entry(e))
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }

            let ext = path
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();

            match ext.as_str() {
                "rs" => {
                    if let Ok(parsed) = RustParser::parse_file(path) {
                        results.push(parsed);
                    }
                }
                "c" | "h" => {
                    if let Ok(parsed) = CParser::parse_file(path) {
                        results.push(parsed);
                    }
                }
                "cc" | "cpp" | "cxx" | "hh" | "hpp" => {
                    if let Ok(parsed) = CppParser::parse_file(path) {
                        results.push(parsed);
                    }
                }
                "md" | "toml" | "json" | "yaml" | "yml" => {
                    let Ok(content) = fs::read_to_string(path) else {
                        continue;
                    };
                    let components = if ext == "md" {
                        re_md_heading
                            .captures_iter(&content)
                            .map(|cap| ComponentInfo {
                                name: cap[2].trim().to_string(),
                                component_type: "chapter_section".to_string(),
                                definition: cap[0].to_string(),
                                functions: Vec::new(),
                            })
                            .collect()
                    } else {
                        Vec::new()
                    };

                    results.push(ParsedFile {
                        path: path.to_string_lossy().to_string(),
                        extension: ext,
                        content,
                        components,
                        global_functions: Vec::new(),
                        includes: Vec::new(),
                    });
                }
                _ => {}
            }
        }

        Ok(results)
    }
}

// ============================================================================
// 1. Parser Rust (`struct`, `enum`, `trait`, `impl`, `fn`, `use`)
// ============================================================================
pub struct RustParser;

impl CodeParser for RustParser {
    fn parse_directory(&self, dir_path: &str) -> Result<Vec<ParsedFile>, Box<dyn Error>> {
        let mut results = Vec::new();
        for entry in WalkDir::new(dir_path)
            .into_iter()
            .filter_entry(|e| !should_skip_entry(e))
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let ext = path
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();

            if ext == "rs" {
                if let Ok(parsed) = Self::parse_file(path) {
                    results.push(parsed);
                }
            } else if matches!(ext.as_str(), "toml" | "md") {
                if let Ok(content) = fs::read_to_string(path) {
                    results.push(ParsedFile {
                        path: path.to_string_lossy().to_string(),
                        extension: ext,
                        content,
                        components: Vec::new(),
                        global_functions: Vec::new(),
                        includes: Vec::new(),
                    });
                }
            }
        }
        Ok(results)
    }
}

impl RustParser {
    pub fn parse_file(path: &Path) -> Result<ParsedFile, Box<dyn Error>> {
        let content = fs::read_to_string(path)?;
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("rs")
            .to_string();

        let re_use = Regex::new(r"(?m)^\s*(?:pub\s+)?use\s+([^;]+);")?;
        let re_item = Regex::new(
            r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?(struct|enum|trait|impl)\b([^{;]*)\{",
        )?;
        let re_fn = Regex::new(
            r#"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?(?:const\s+)?(?:extern\s+(?:"[^"]+"\s+)?)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^>]*>)?\s*\([^;{}]*\)(?:\s*->\s*[^;{]+)?\s*[;{]"#,
        )?;

        let includes: Vec<String> = re_use
            .captures_iter(&content)
            .map(|cap| cap[1].trim().to_string())
            .collect();

        let mut components = Vec::new();
        let mut block_ranges: Vec<Range<usize>> = Vec::new();

        for cap in re_item.captures_iter(&content) {
            let full_match = cap.get(0).unwrap();
            let kind = &cap[1]; // "struct", "enum", "trait", "impl"
            let header_tail = cap[2].trim();
            let opening_brace = full_match.end() - 1;

            let Some(closing_brace) = find_matching_brace(&content, opening_brace) else {
                continue;
            };
            let definition_end = closing_brace + 1;
            block_ranges.push(full_match.start()..definition_end);

            let definition = content[full_match.start()..definition_end].to_string();
            let body = &content[opening_brace + 1..closing_brace];

            let name = if kind == "impl" {
                let target = header_tail
                    .split(" for ")
                    .last()
                    .unwrap_or(header_tail)
                    .split('<')
                    .next()
                    .unwrap_or(header_tail)
                    .trim();
                if target.is_empty() {
                    "impl".to_string()
                } else {
                    format!("impl {}", target)
                }
            } else {
                header_tail
                    .split(['<', '(', ':'])
                    .next()
                    .unwrap_or(header_tail)
                    .split_whitespace()
                    .next()
                    .unwrap_or(kind)
                    .to_string()
            };

            let functions = if matches!(kind, "impl" | "trait") {
                let clean_body = strip_function_bodies(body);
                extract_rust_functions(&re_fn, &clean_body)
            } else {
                Vec::new()
            };

            components.push(ComponentInfo {
                name,
                component_type: kind.to_string(),
                definition,
                functions,
            });
        }

        let top_level = strip_ranges(&content, &block_ranges);
        let clean_top_level = strip_function_bodies(&top_level);
        let global_functions = extract_rust_functions(&re_fn, &clean_top_level);

        Ok(ParsedFile {
            path: path.to_string_lossy().to_string(),
            extension: ext,
            content,
            components,
            global_functions,
            includes,
        })
    }
}

fn extract_rust_functions(re_fn: &Regex, source: &str) -> Vec<FunctionInfo> {
    let mut functions = Vec::new();
    for cap in re_fn.captures_iter(source) {
        let signature = cap
            .get(0)
            .unwrap()
            .as_str()
            .trim()
            .trim_end_matches([';', '{'])
            .trim()
            .to_string();
        let fn_name = cap[1].to_string();
        functions.push(FunctionInfo {
            description: format!("Funcția Rust {}", fn_name),
            name: fn_name,
            signature,
        });
    }
    functions
}

// ============================================================================
// 2. Parser C pur (struct, typedef struct, enum, typedef enum, funcții globale)
// ============================================================================
pub struct CParser;

impl CodeParser for CParser {
    fn parse_directory(&self, dir_path: &str) -> Result<Vec<ParsedFile>, Box<dyn Error>> {
        Self::parse_directory_internal(dir_path)
    }
}

impl CParser {
    pub fn parse_file(path: &Path) -> Result<ParsedFile, Box<dyn Error>> {
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        let content = fs::read_to_string(path)?;

        let re_type_block = Regex::new(
            r"(?m)\b(typedef\s+)?(struct|enum)(?:\s+([A-Za-z_][A-Za-z0-9_]*))?\s*\{",
        )?;
        let re_typedef_tail = Regex::new(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*;")?;
        let re_c_func = Regex::new(
            r"(?m)^\s*(?:(?:static|inline|extern|const|volatile|struct|enum|unsigned|signed|long|short)\s+)*(?:[A-Za-z_][A-Za-z0-9_]*\s*[*&\s]+)+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*[;{]",
        )?;
        let re_include = Regex::new(r#"#include\s+[<"]([^>"]+)[>"]"#)?;

        let includes: Vec<String> = re_include
            .captures_iter(&content)
            .map(|cap| cap[1].to_string())
            .collect();

        let mut components = Vec::new();
        let mut global_functions = Vec::new();
        let mut block_ranges: Vec<Range<usize>> = Vec::new();

        if ext != "md" {
            for cap in re_type_block.captures_iter(&content) {
                let full_match = cap.get(0).unwrap();
                let is_typedef = cap.get(1).is_some();
                let kind = &cap[2];
                let tag_name = cap.get(3).map(|m| m.as_str().to_string());

                let opening_brace = full_match.end() - 1;
                let Some(closing_brace) = find_matching_brace(&content, opening_brace) else {
                    continue;
                };

                let after_brace = &content[closing_brace + 1..];
                let mut definition_end = closing_brace + 1;
                let mut resolved_name = tag_name.clone();

                if is_typedef {
                    if let Some(tail_cap) = re_typedef_tail.captures(after_brace) {
                        resolved_name = Some(tail_cap[1].to_string());
                        definition_end += tail_cap.get(0).unwrap().end();
                    }
                } else if after_brace.as_bytes().first() == Some(&b';') {
                    definition_end += 1;
                }

                let Some(name) = resolved_name else {
                    continue;
                };

                block_ranges.push(full_match.start()..definition_end);
                let definition = content[full_match.start()..definition_end].to_string();

                components.push(ComponentInfo {
                    name,
                    component_type: kind.to_string(),
                    definition,
                    functions: Vec::new(),
                });
            }

            let outside_types = strip_ranges(&content, &block_ranges);
            let clean_source = strip_function_bodies(&outside_types);

            for cap_func in re_c_func.captures_iter(&clean_source) {
                let full_sig = cap_func
                    .get(0)
                    .unwrap()
                    .as_str()
                    .trim()
                    .trim_end_matches([';', '{'])
                    .trim()
                    .to_string();
                let fn_name = cap_func[1].to_string();

                if matches!(
                    fn_name.as_str(),
                    "if" | "while" | "for" | "switch" | "return" | "sizeof"
                ) {
                    continue;
                }

                global_functions.push(FunctionInfo {
                    description: format!("Funcția C {}", fn_name),
                    name: fn_name,
                    signature: full_sig,
                });
            }
        }

        Ok(ParsedFile {
            path: path.to_string_lossy().to_string(),
            extension: ext.to_string(),
            content,
            components,
            global_functions,
            includes,
        })
    }

    pub fn parse_directory_internal(dir_path: &str) -> Result<Vec<ParsedFile>, Box<dyn Error>> {
        let mut results = Vec::new();
        for entry in WalkDir::new(dir_path)
            .into_iter()
            .filter_entry(|entry| !should_skip_entry(entry))
            .filter_map(|entry| entry.ok())
        {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
            if !matches!(ext, "c" | "h" | "md") {
                continue;
            }
            if let Ok(parsed) = Self::parse_file(path) {
                results.push(parsed);
            }
        }
        Ok(results)
    }
}

// ============================================================================
// 3. Parser C++ (extrage clase, struct-uri, metode ȘI funcții globale)
// ============================================================================
pub struct CppParser;

impl CodeParser for CppParser {
    fn parse_directory(&self, dir_path: &str) -> Result<Vec<ParsedFile>, Box<dyn Error>> {
        Self::parse_directory_internal(dir_path)
    }
}

impl CppParser {
    pub fn parse_file(path: &Path) -> Result<ParsedFile, Box<dyn Error>> {
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        let content = fs::read_to_string(path)?;

        let re_class = Regex::new(r"(?:class|struct)\s+([A-Za-z_][A-Za-z0-9_]*)\b[^{;]*\{")?;
        let re_func = Regex::new(
            r"((?:[A-Za-z_][A-Za-z0-9_<>,:\s*&]*?\s+)?(?:~?[A-Za-z_][A-Za-z0-9_:~]*)\s*\([^;{}]*\)\s*(?:(?:const|volatile)\s*)?(?:&&?|&)?\s*(?:noexcept(?:\s*\([^)]*\))?\s*)?(?:->\s*[^;{}]+)?\s*(?:override|final)?\s*(?:=\s*(?:0|default|delete))?)\s*[;{]",
        )?;
        let re_access = Regex::new(r"\b(?:public|private|protected)\s*:")?;
        let re_include = Regex::new(r#"#include\s+[<"]([^>"]+)[>"]"#)?;

        let includes: Vec<String> = re_include
            .captures_iter(&content)
            .map(|cap| cap[1].to_string())
            .collect();

        let mut components = Vec::new();
        let mut global_functions = Vec::new();
        let mut class_ranges: Vec<Range<usize>> = Vec::new();

        if ext != "md" {
            for cap_class in re_class.captures_iter(&content) {
                let full_header = cap_class.get(0).unwrap().as_str();
                let comp_type = if full_header.trim_start().starts_with("struct") {
                    "struct"
                } else {
                    "class"
                };
                let class_name = cap_class[1].to_string();
                let class_match = cap_class.get(0).unwrap();
                let opening_brace = class_match.end() - 1;

                let Some(closing_brace) = find_matching_brace(&content, opening_brace) else {
                    continue;
                };
                let mut definition_end = closing_brace + 1;
                if content.as_bytes().get(definition_end) == Some(&b';') {
                    definition_end += 1;
                }

                class_ranges.push(class_match.start()..definition_end);

                let definition = content[class_match.start()..definition_end].to_string();
                let class_body = &content[opening_brace + 1..closing_brace];
                let member_source = re_access.replace_all(class_body, " ");
                let clean_member_source = strip_function_bodies(&member_source);
                let functions = extract_functions(&re_func, &clean_member_source);

                components.push(ComponentInfo {
                    name: class_name,
                    component_type: comp_type.to_string(),
                    definition,
                    functions,
                });
            }

            let top_level_code = strip_ranges(&content, &class_ranges);
            let clean_top_level = strip_function_bodies(&top_level_code);
            global_functions = extract_functions(&re_func, &clean_top_level);
        }

        Ok(ParsedFile {
            path: path.to_string_lossy().to_string(),
            extension: ext.to_string(),
            content,
            components,
            global_functions,
            includes,
        })
    }

    pub fn parse_directory_internal(dir_path: &str) -> Result<Vec<ParsedFile>, Box<dyn Error>> {
        let mut results = Vec::new();
        for entry in WalkDir::new(dir_path)
            .into_iter()
            .filter_entry(|entry| !should_skip_entry(entry))
            .filter_map(|entry| entry.ok())
        {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
            if !matches!(ext, "c" | "cc" | "cpp" | "cxx" | "h" | "hh" | "hpp" | "md") {
                continue;
            }
            if let Ok(parsed) = Self::parse_file(path) {
                results.push(parsed);
            }
        }
        Ok(results)
    }
}

// ============================================================================
// 4. Parser Markdown / Cărți (extrage secțiunile `#` și `##` ca componente)
// ============================================================================
pub struct MarkdownParser;

impl CodeParser for MarkdownParser {
    fn parse_directory(&self, dir_path: &str) -> Result<Vec<ParsedFile>, Box<dyn Error>> {
        let mut results = Vec::new();
        let re_heading = Regex::new(r"(?m)^(#{1,3})\s+(.+)$")?;

        for entry in WalkDir::new(dir_path)
            .into_iter()
            .filter_entry(|entry| !should_skip_entry(entry))
            .filter_map(|entry| entry.ok())
        {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
            if !matches!(ext, "md" | "txt") {
                continue;
            }

            let content = match fs::read_to_string(path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let components: Vec<ComponentInfo> = re_heading
                .captures_iter(&content)
                .map(|cap| ComponentInfo {
                    name: cap[2].trim().to_string(),
                    component_type: "chapter_section".to_string(),
                    definition: cap[0].to_string(),
                    functions: Vec::new(),
                })
                .collect();

            results.push(ParsedFile {
                path: path.to_string_lossy().to_string(),
                extension: ext.to_string(),
                content,
                components,
                global_functions: Vec::new(),
                includes: Vec::new(),
            });
        }

        Ok(results)
    }
}

// ============================================================================
// Helpers de parsare
// ============================================================================
fn extract_functions(re_func: &Regex, source: &str) -> Vec<FunctionInfo> {
    let mut functions = Vec::new();
    for cap_func in re_func.captures_iter(source) {
        let signature = cap_func[1].trim().to_string();
        let fn_name = signature
            .split('(')
            .next()
            .unwrap_or("")
            .split_whitespace()
            .last()
            .unwrap_or("unknown")
            .trim_start_matches('*')
            .to_string();

        if matches!(
            fn_name.as_str(),
            "if" | "while" | "for" | "switch" | "return" | "sizeof"
        ) {
            continue;
        }

        functions.push(FunctionInfo {
            description: format!("Funcția {}", fn_name),
            name: fn_name,
            signature,
        });
    }
    functions
}

/// Înlocuiește blocurile `{ ... }` ale funcțiilor cu `;` pentru a păstra strict prototipurile/anteturile
fn strip_function_bodies(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let bytes = source.as_bytes();
    let mut i = 0;
    let mut state = LexState::Code;

    while i < bytes.len() {
        match state {
            LexState::Code => match bytes[i] {
                b'R' if bytes.get(i + 1) == Some(&b'"') => {
                    if let Some(end) = find_raw_string_end(bytes, i) {
                        out.push_str(&source[i..=end]);
                        i = end + 1;
                        continue;
                    }
                }
                b'/' if bytes.get(i + 1) == Some(&b'/') => {
                    state = LexState::LineComment;
                    out.push_str("//");
                    i += 2;
                    continue;
                }
                b'/' if bytes.get(i + 1) == Some(&b'*') => {
                    state = LexState::BlockComment;
                    out.push_str("/*");
                    i += 2;
                    continue;
                }
                b'"' => {
                    state = LexState::String;
                    out.push('"');
                }
                b'\'' => {
                    state = LexState::Character;
                    out.push('\'');
                }
                b'{' => {
                    if let Some(close) = find_matching_brace(source, i) {
                        out.push(';');
                        i = close + 1;
                        continue;
                    }
                    out.push('{');
                }
                _ => out.push(bytes[i] as char),
            },
            LexState::LineComment => {
                out.push(bytes[i] as char);
                if bytes[i] == b'\n' {
                    state = LexState::Code;
                }
            }
            LexState::BlockComment => {
                if bytes[i] == b'*' && bytes.get(i + 1) == Some(&b'/') {
                    out.push_str("*/");
                    state = LexState::Code;
                    i += 2;
                    continue;
                }
                out.push(bytes[i] as char);
            }
            LexState::String => {
                out.push(bytes[i] as char);
                if bytes[i] == b'\\' && i + 1 < bytes.len() {
                    i += 1;
                    out.push(bytes[i] as char);
                } else if bytes[i] == b'"' {
                    state = LexState::Code;
                }
            }
            LexState::Character => {
                out.push(bytes[i] as char);
                if bytes[i] == b'\\' && i + 1 < bytes.len() {
                    i += 1;
                    out.push(bytes[i] as char);
                } else if bytes[i] == b'\'' {
                    state = LexState::Code;
                }
            }
        }
        i += 1;
    }
    out
}

/// Înlocuiește blocurile de clase/struct-uri cu spații, ca să extragem doar funcțiile globale
fn strip_ranges(source: &str, ranges: &[Range<usize>]) -> String {
    if ranges.is_empty() {
        return source.to_string();
    }
    let mut out = String::with_capacity(source.len());
    let mut cursor = 0;
    for r in ranges {
        if r.start >= cursor {
            out.push_str(&source[cursor..r.start]);
            out.push(' ');
            cursor = r.end;
        }
    }
    if cursor < source.len() {
        out.push_str(&source[cursor..]);
    }
    out
}

fn find_matching_brace(source: &str, opening_brace: usize) -> Option<usize> {
    let bytes = source.as_bytes();
    let mut depth = 0usize;
    let mut index = opening_brace;
    let mut state = LexState::Code;

    while index < bytes.len() {
        match state {
            LexState::Code => match bytes[index] {
                b'R' if bytes.get(index + 1) == Some(&b'"') => {
                    index = find_raw_string_end(bytes, index)?;
                }
                b'/' if bytes.get(index + 1) == Some(&b'/') => {
                    state = LexState::LineComment;
                    index += 1;
                }
                b'/' if bytes.get(index + 1) == Some(&b'*') => {
                    state = LexState::BlockComment;
                    index += 1;
                }
                b'"' => state = LexState::String,
                b'\'' => state = LexState::Character,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(index);
                    }
                }
                _ => {}
            },
            LexState::LineComment if bytes[index] == b'\n' => state = LexState::Code,
            LexState::BlockComment
                if bytes[index] == b'*' && bytes.get(index + 1) == Some(&b'/') =>
            {
                state = LexState::Code;
                index += 1;
            }
            LexState::String | LexState::Character if bytes[index] == b'\\' => index += 1,
            LexState::String if bytes[index] == b'"' => state = LexState::Code,
            LexState::Character if bytes[index] == b'\'' => state = LexState::Code,
            _ => {}
        }
        index += 1;
    }

    None
}

fn find_raw_string_end(bytes: &[u8], start: usize) -> Option<usize> {
    let delimiter_start = start + 2;
    let open_paren = (delimiter_start..bytes.len().min(delimiter_start + 17))
        .find(|&index| bytes[index] == b'(')?;
    let delimiter = &bytes[delimiter_start..open_paren];
    let terminator = [b")".as_slice(), delimiter, b"\""].concat();
    let terminator_start = bytes[open_paren + 1..]
        .windows(terminator.len())
        .position(|window| window == terminator)?
        + open_paren
        + 1;
    Some(terminator_start + terminator.len() - 1)
}

enum LexState {
    Code,
    LineComment,
    BlockComment,
    String,
    Character,
}

#[cfg(test)]
mod tests {
    use super::{CParser, CodeParser, CppParser, PolyglotCodeParser, RustParser};
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn rust_parser_extracts_structs_enums_traits_impls_and_functions() {
        let unique_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("cortex-rust-parser-{unique_id}"));
        fs::create_dir_all(directory.join("src")).expect("temporary directory should be created");
        fs::create_dir_all(directory.join("target")).expect("target directory should be created");

        let source = r#"
use std::io::Result;

pub enum Mode {
    Normal,
    Insert,
}

pub struct Editor {
    pub cursor: usize,
}

pub trait Renderable {
    fn render(&self);
}

impl Editor {
    pub fn new() -> Self {
        Self { cursor: 0 }
    }

    pub async fn refresh(&mut self) -> Result<()> {
        Ok(())
    }
}

pub fn run_app() {
    println!("running");
}
"#;
        fs::write(directory.join("src").join("main.rs"), source).expect("Rust source written");
        fs::write(directory.join("Cargo.toml"), "[package]\nname = \"demo\"\n")
            .expect("Cargo.toml written");
        // Fișierele din `target/` trebuie ignorate automat
        fs::write(directory.join("target").join("ignored.rs"), "pub fn ignored() {}")
            .expect("target file written");

        let files = PolyglotCodeParser
            .parse_directory(directory.to_str().expect("path should be valid"))
            .expect("project should parse");

        assert_eq!(files.len(), 2); // src/main.rs și Cargo.toml
        let rs_file = files
            .iter()
            .find(|f| f.extension == "rs")
            .expect("main.rs should be parsed");

        assert_eq!(rs_file.includes, vec!["std::io::Result"]);
        assert!(rs_file.components.iter().any(|c| c.name == "Mode" && c.component_type == "enum"));
        assert!(rs_file.components.iter().any(|c| c.name == "Editor" && c.component_type == "struct"));
        assert!(rs_file.components.iter().any(|c| c.name == "Renderable" && c.component_type == "trait"));

        let impl_comp = rs_file
            .components
            .iter()
            .find(|c| c.name == "impl Editor")
            .expect("impl Editor should be extracted");
        assert!(impl_comp.functions.iter().any(|f| f.name == "new"));
        assert!(impl_comp.functions.iter().any(|f| f.name == "refresh"));

        assert_eq!(rs_file.global_functions.len(), 1);
        assert_eq!(rs_file.global_functions[0].name, "run_app");

        // Verificăm și RustParser direct
        let direct_rust = RustParser
            .parse_directory(directory.to_str().expect("path should be valid"))
            .expect("RustParser should parse");
        assert_eq!(direct_rust.len(), 2);

        fs::remove_dir_all(directory).expect("temporary directory should be removed");
    }

    #[test]
    fn c_parser_extracts_typedef_structs_enums_and_global_functions() {
        let unique_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("cortex-c-parser-{unique_id}"));
        fs::create_dir_all(&directory).expect("temporary directory should be created");

        let header = r#"
typedef enum {
    MODE_NORMAL,
    MODE_INSERT
} EditorMode;

typedef struct {
    int cursor_x;
    int cursor_y;
    EditorMode mode;
} EditorState;

struct Buffer {
    char *data;
};

static inline char* get_current_line(EditorState *state);
void init_editor(EditorState *state) {
    printf("Init\n");
    state->cursor_x = 0;
}
"#;
        fs::write(directory.join("editor.h"), header).expect("C header should be written");

        let files = CParser
            .parse_directory(directory.to_str().expect("path should be valid"))
            .expect("C header should parse");

        assert_eq!(files.len(), 1);
        assert_eq!(files[0].components.len(), 3);

        assert_eq!(files[0].components[0].name, "EditorMode");
        assert_eq!(files[0].components[0].component_type, "enum");

        assert_eq!(files[0].components[1].name, "EditorState");
        assert_eq!(files[0].components[1].component_type, "struct");

        assert_eq!(files[0].components[2].name, "Buffer");
        assert_eq!(files[0].components[2].component_type, "struct");

        assert_eq!(files[0].global_functions.len(), 2);
        assert!(files[0]
            .global_functions
            .iter()
            .any(|f| f.name == "get_current_line"));
        assert!(files[0]
            .global_functions
            .iter()
            .any(|f| f.name == "init_editor"));
        assert!(files[0]
            .global_functions
            .iter()
            .all(|f| f.name != "printf"));

        fs::remove_dir_all(directory).expect("temporary directory should be removed");
    }

    #[test]
    fn extracts_class_definitions_and_only_their_function_prototypes() {
        let unique_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("cortex-parser-{unique_id}"));
        fs::create_dir_all(&directory).expect("temporary directory should be created");
        let source = r#"
class First : public Base {
public:
    int get_value(int fallback) const;
    void set_value(int value) { auto marker = R"tag(})tag"; helper_call(value); value_ = value; }
private:
    int value_;
};

struct Second {
    bool ready() noexcept override;
};
"#;
        fs::write(directory.join("sample.hpp"), source).expect("sample source should be written");

        let files = CppParser
            .parse_directory(directory.to_str().expect("path should be valid"))
            .expect("source should parse");

        assert_eq!(files.len(), 1);
        assert_eq!(files[0].components.len(), 2);
        assert_eq!(files[0].components[0].name, "First");
        assert!(files[0].components[0]
            .definition
            .contains("int get_value(int fallback) const;"));
        assert!(files[0].components[0]
            .definition
            .contains("value_ = value; }"));
        assert!(files[0].components[0]
            .functions
            .iter()
            .any(|function| function.signature == "int get_value(int fallback) const"));
        assert!(files[0].components[0]
            .functions
            .iter()
            .all(|function| function.name != "helper_call" && function.name != "ready"));
        assert_eq!(files[0].components[1].name, "Second");
        assert_eq!(files[0].components[1].component_type, "struct");
        assert!(files[0].components[1]
            .functions
            .iter()
            .any(|function| function.signature == "bool ready() noexcept override"));

        fs::remove_dir_all(directory).expect("temporary directory should be removed");
    }
}