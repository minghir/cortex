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