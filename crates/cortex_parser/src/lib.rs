use regex::Regex;
use std::error::Error;
use std::fs;
use walkdir::WalkDir;

#[derive(Debug, Clone)]
pub struct FunctionInfo {
    pub name: String,
    pub signature: String,
}

#[derive(Debug, Clone)]
pub struct ClassInfo {
    pub name: String,
    pub functions: Vec<FunctionInfo>,
}

#[derive(Debug, Clone)]
pub struct ParsedCppFile {
    pub path: String,
    pub extension: String,
    pub content: String,
    pub classes: Vec<ClassInfo>,
    pub includes: Vec<String>,
}

pub struct CppParser;

impl CppParser {
    pub fn parse_directory(dir_path: &str) -> Result<Vec<ParsedCppFile>, Box<dyn Error>> {
        let mut results = Vec::new();
        let re_class = Regex::new(r"(?:class|struct)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{")?;
        let re_func = Regex::new(r"([A-Za-z_][A-Za-z0-9_<>\:\*&]*\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^;\{]*\))\s*[\{;]")?;
        let re_include = Regex::new(r#"#include\s+[<"]([^>"]+)[>"]"#)?;

        for entry in WalkDir::new(dir_path).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }

            let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
            if ext != "cpp" && ext != "h" && ext != "hpp" && ext != "cxx" {
                continue;
            }

            let content = match fs::read_to_string(path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let includes: Vec<String> = re_include
                .captures_iter(&content)
                .map(|cap| cap[1].to_string())
                .collect();

            let mut classes = Vec::new();
            for cap_class in re_class.captures_iter(&content) {
                let class_name = cap_class[1].to_string();
                let mut functions = Vec::new();

                for cap_func in re_func.captures_iter(&content) {
                    let signature = cap_func[1].trim().to_string();
                    let fn_name = signature
                        .split('(')
                        .next()
                        .unwrap_or("")
                        .split_whitespace()
                        .last()
                        .unwrap_or("unknown")
                        .to_string();

                    if matches!(fn_name.as_str(), "if" | "while" | "for" | "switch") {
                        continue;
                    }

                    functions.push(FunctionInfo {
                        name: fn_name,
                        signature,
                    });
                }

                classes.push(ClassInfo {
                    name: class_name,
                    functions,
                });
            }

            results.push(ParsedCppFile {
                path: path.to_string_lossy().to_string(),
                extension: ext.to_string(),
                content,
                classes,
                includes,
            });
        }

        Ok(results)
    }
}