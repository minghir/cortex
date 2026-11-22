use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectDto {
    pub id: i64,
    pub name: String,
    pub project_type: String,
    pub root_path: Option<String>,
    pub git_remote: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteFilePayload {
    pub relative_path: String,
    pub content: String,
    #[serde(default)]
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsageDto {
    pub prompt_tokens: Option<u32>,
    pub candidates_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatHistoryEntryDto {
    pub role: String,
    pub content: String,
}