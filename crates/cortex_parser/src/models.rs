use std::error::Error;

#[derive(Debug, Clone)]
pub struct FunctionInfo {
    pub name: String,
    pub signature: String,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct ComponentInfo {
    pub name: String,
    pub component_type: String, // "class", "struct", "module", "chapter_section"
    pub definition: String,
    pub functions: Vec<FunctionInfo>,
}

#[derive(Debug, Clone)]
pub struct ParsedFile {
    pub path: String,
    pub extension: String,
    pub content: String,
    pub components: Vec<ComponentInfo>,
    pub global_functions: Vec<FunctionInfo>,
    pub includes: Vec<String>,
}

/// Trait comun pe care îl implementează orice parser (C, C++, Python, oli, Markdown)
pub trait CodeParser: Send + Sync {
    fn parse_directory(&self, dir_path: &str) -> Result<Vec<ParsedFile>, Box<dyn Error>>;
}