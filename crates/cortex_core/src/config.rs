pub(crate) fn ollama_url(path: &str) -> String {
    let base_url =
        std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
    format!(
        "{}/{}",
        base_url.trim_end_matches('/'),
        path.trim_start_matches('/')
    )
}

pub(crate) fn ollama_chat_model() -> String {
    std::env::var("OLLAMA_CHAT_MODEL").unwrap_or_else(|_| "llama3.2".to_string())
}


pub fn gemini_api_key() -> Result<String, std::env::VarError> {
    if let Ok(key) = std::env::var("GEMINI_API_KEY") {
        let trimmed = key.trim().to_string();
        if !trimmed.is_empty() {
            return Ok(trimmed);
        }
    }

    // Fallback: reîncarcă .env dacă a fost editat după pornirea aplicației
    let _ = dotenvy::dotenv();
    std::env::var("GEMINI_API_KEY").map(|k| k.trim().to_string())
}

pub fn gemini_chat_model() -> String {
    std::env::var("GEMINI_MODEL").unwrap_or_else(|_| "gemini-3.8-flash".to_string())
}