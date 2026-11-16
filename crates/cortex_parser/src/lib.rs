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
    pub definition: String,
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
        let re_class = Regex::new(r"(?:class|struct)\s+([A-Za-z_][A-Za-z0-9_]*)\b[^{;]*\{")?;
        let re_func = Regex::new(
            r"((?:[A-Za-z_][A-Za-z0-9_<>,:\s*&]*?\s+)?(?:~?[A-Za-z_][A-Za-z0-9_:~]*)\s*\([^;{}]*\)\s*(?:(?:const|volatile)\s*)?(?:&&?|&)?\s*(?:noexcept(?:\s*\([^)]*\))?\s*)?(?:->\s*[^;{}]+)?\s*(?:override|final)?\s*(?:=\s*(?:0|default|delete))?)\s*[;{]",
        )?;
        let re_access = Regex::new(r"\b(?:public|private|protected)\s*:")?;
        let re_include = Regex::new(r#"#include\s+[<"]([^>"]+)[>"]"#)?;

        for entry in WalkDir::new(dir_path)
            .into_iter()
            .filter_entry(|entry| entry.file_name() != ".git")
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

            let content = match fs::read_to_string(path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let includes: Vec<String> = re_include
                .captures_iter(&content)
                .map(|cap| cap[1].to_string())
                .collect();

            let mut classes = Vec::new();
            if ext != "md" {
                for cap_class in re_class.captures_iter(&content) {
                    let class_name = cap_class[1].to_string();
                    let class_match = cap_class
                        .get(0)
                        .expect("class regex always has a full match");
                    let opening_brace = class_match.end() - 1;
                    let Some(closing_brace) = find_matching_brace(&content, opening_brace) else {
                        continue;
                    };
                    let mut definition_end = closing_brace + 1;
                    if content.as_bytes().get(definition_end) == Some(&b';') {
                        definition_end += 1;
                    }
                    let definition = content[class_match.start()..definition_end].to_string();
                    let class_body = &content[opening_brace + 1..closing_brace];
                    let member_source = re_access.replace_all(class_body, " ");
                    let mut functions = Vec::new();

                    for cap_func in re_func.captures_iter(&member_source) {
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
                        definition,
                    });
                }
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
    use super::CppParser;
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

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
    void set_value(int value) { auto marker = R"tag(})tag"; value_ = value; }
private:
    int value_;
};

struct Second {
    bool ready() noexcept override;
};
"#;
        fs::write(directory.join("sample.hpp"), source).expect("sample source should be written");

        let files = CppParser::parse_directory(directory.to_str().expect("path should be valid"))
            .expect("source should parse");

        assert_eq!(files.len(), 1);
        assert_eq!(files[0].classes.len(), 2);
        assert_eq!(files[0].classes[0].name, "First");
        assert!(files[0].classes[0]
            .definition
            .contains("int get_value(int fallback) const;"));
        assert!(files[0].classes[0].definition.contains("value_ = value; }"));
        assert!(files[0].classes[0]
            .functions
            .iter()
            .any(|function| function.signature == "int get_value(int fallback) const"));
        assert!(files[0].classes[0]
            .functions
            .iter()
            .all(|function| function.name != "ready"));
        assert_eq!(files[0].classes[1].name, "Second");
        assert!(files[0].classes[1]
            .functions
            .iter()
            .any(|function| function.signature == "bool ready() noexcept override"));

        fs::remove_dir_all(directory).expect("temporary directory should be removed");
    }

    #[test]
    fn includes_c_and_markdown_files_but_does_not_parse_markdown_as_code() {
        let unique_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("cortex-parser-{unique_id}"));
        fs::create_dir_all(&directory).expect("temporary directory should be created");
        fs::write(directory.join("main.c"), "int main(void) { return 0; }")
            .expect("C source should be written");
        fs::write(
            directory.join("README.md"),
            "Example: class Widget { void run(); };",
        )
        .expect("documentation should be written");

        let files = CppParser::parse_directory(directory.to_str().expect("path should be valid"))
            .expect("source and documentation should parse");

        assert_eq!(files.len(), 2);
        assert!(files.iter().any(|file| file.extension == "c"));
        let markdown = files
            .iter()
            .find(|file| file.extension == "md")
            .expect("markdown should be indexed");
        assert!(markdown.classes.is_empty());

        fs::remove_dir_all(directory).expect("temporary directory should be removed");
    }
}
