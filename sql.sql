-- Activăm extensia pgvector dacă nu este deja activată
CREATE EXTENSION IF NOT EXISTS vector;

-- Atenție: șterge toate tabelele și datele din schema brain.
DROP SCHEMA IF EXISTS brain CASCADE;

-- 1. Creare schema brain
CREATE SCHEMA IF NOT EXISTS brain;

-- 2. Proiecte (Suportă orice tip de proiect: cod C, C++, Python, oli, Web sau Scriere Carte)
CREATE TABLE brain.projects (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    project_type VARCHAR(50) NOT NULL, -- ex: 'c_code', 'cpp_code', 'oli_language', 'book', 'documentation'
    status VARCHAR(32) NOT NULL DEFAULT 'active', -- 'active', 'archived', 'completed'
    description TEXT,
    tech_stack JSONB DEFAULT '[]'::jsonb, -- ex: ["c", "oli", "postgresql"]
    root_path TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 3. Etichete (Tags) pentru organizare flexibilă
CREATE TABLE brain.tags (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(64) NOT NULL UNIQUE
);

CREATE TABLE brain.project_tags (
    project_id BIGINT NOT NULL REFERENCES brain.projects(id) ON DELETE CASCADE,
    tag_id BIGINT NOT NULL REFERENCES brain.tags(id) ON DELETE CASCADE,
    PRIMARY KEY (project_id, tag_id)
);

-- 4. Decizii de Arhitectură / Note de Design / Structură Carte
CREATE TABLE brain.architecture_decisions (
    id BIGSERIAL PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES brain.projects(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    reason TEXT NOT NULL,
    decision TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 5. Fișiere sursă / Capitole (Suportă ierarhie prin parent_id pentru foldere sau sub-capitole)
CREATE TABLE brain.source_files (
    id BIGSERIAL PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES brain.projects(id) ON DELETE CASCADE,
    parent_id BIGINT REFERENCES brain.source_files(id) ON DELETE CASCADE, -- Pentru ierarhii (folder/sub-capitol)
    path TEXT NOT NULL,
    extension VARCHAR(32), -- ex: 'c', 'h', 'cpp', 'oli', 'md', 'txt'
    content TEXT,
    title VARCHAR(255), -- Titlu prietenos (util pentru capitole de carte sau module)
    summary TEXT,
    language VARCHAR(50), -- ex: 'c', 'cpp', 'oli', 'markdown', 'python', 'text'
    status VARCHAR(32) DEFAULT 'draft', -- 'draft', 'review', 'final'
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_path_per_project UNIQUE(project_id, path)
);

-- 6. Componente de cod sau Secțiuni din carte (Polivalent)
-- Poate stoca: Struct-uri în C, clase în C++, module în 'oli', sau capitole/secțiuni dintr-o carte
CREATE TABLE brain.code_components (
    id BIGSERIAL PRIMARY KEY,
    file_id BIGINT NOT NULL REFERENCES brain.source_files(id) ON DELETE CASCADE,
    component_type VARCHAR(50) NOT NULL, -- ex: 'struct', 'enum', 'class', 'oli_module', 'chapter_section', 'concept'
    name VARCHAR(255) NOT NULL,
    description TEXT,
    definition TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 7. Funcții (Pentru cod) sau Blocuri de text/paragrafe importante
CREATE TABLE brain.functions (
    id BIGSERIAL PRIMARY KEY,
    file_id BIGINT NOT NULL REFERENCES brain.source_files(id) ON DELETE CASCADE,
    component_id BIGINT REFERENCES brain.code_components(id) ON DELETE SET NULL,
    name VARCHAR(255) NOT NULL,
    signature TEXT,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 8. Biblioteca de Prompt-uri (Pentru a salva interacțiunile bune cu AI-ul)
CREATE TABLE brain.prompts_library (
    id BIGSERIAL PRIMARY KEY,
    project_id BIGINT REFERENCES brain.projects(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    prompt_text TEXT NOT NULL,
    response_summary TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 9. Tabela "BRAIN" (Memories / RAG cu suport pentru multiple modele de embeddings)
CREATE TABLE brain.memories (
    id BIGSERIAL PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES brain.projects(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    content_type VARCHAR(50) NOT NULL, -- 'source_code', 'test_code', 'documentation', 'function', 'architecture'
    source_table VARCHAR(50),          -- Numele tabelei sursă (ex: 'source_files', 'functions', 'architecture_decisions')
    source_id BIGINT,                  -- ID-ul din tabela sursă
    
    -- Suport dual pentru embeddings (poți folosi unul sau pe ambele în paralel)
    embedding_768 VECTOR(768),         -- Pentru Google Gemini (text-embedding-004) sau Ollama (nomic-embed-text)
    embedding_1536 VECTOR(1536),       -- Pentru un model alternativ (ex: OpenAI text-embedding-3-small)
    
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Indexuri pentru performanță și căutare semantică rapidă (HNSW)
CREATE INDEX idx_memories_project_id ON brain.memories(project_id);

CREATE INDEX idx_memories_embedding_768 ON brain.memories 
USING hnsw (embedding_768 vector_cosine_ops);

CREATE INDEX idx_memories_embedding_1536 ON brain.memories 
USING hnsw (embedding_1536 vector_cosine_ops);

-- 10. Istoricul conversațiilor separat pe proiect
CREATE TABLE brain.chat_history (
    id BIGSERIAL PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES brain.projects(id) ON DELETE CASCADE,
    role VARCHAR(16) NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_chat_history_project_id ON brain.chat_history(project_id, id DESC);