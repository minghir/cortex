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
