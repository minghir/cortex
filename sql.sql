-- 1. Creare schema brain
CREATE SCHEMA IF NOT EXISTS brain;

-- 2. Proiecte
CREATE TABLE brain.projects (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    tech_stack JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 3. Decizii de Arhitectură
CREATE TABLE brain.architecture_decisions (
    id BIGSERIAL PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES brain.projects(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    reason TEXT NOT NULL,
    decision TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 4. Fișiere sursă
CREATE TABLE brain.source_files (
    id BIGSERIAL PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES brain.projects(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    summary TEXT,
    language VARCHAR(50),
    CONSTRAINT unique_path_per_project UNIQUE(project_id, path)
);

-- 5. Clase
CREATE TABLE brain.classes (
    id BIGSERIAL PRIMARY KEY,
    file_id BIGINT NOT NULL REFERENCES brain.source_files(id) ON DELETE CASCADE,
    class_name VARCHAR(255) NOT NULL,
    description TEXT
);

-- 6. Funcții
CREATE TABLE brain.functions (
    id BIGSERIAL PRIMARY KEY,
    class_id BIGINT NOT NULL REFERENCES brain.classes(id) ON DELETE CASCADE,
    function_name VARCHAR(255) NOT NULL,
    signature TEXT,
    description TEXT
);

-- 7. Tabela "BRAIN" (Memories)
CREATE TABLE brain.memories (
    id BIGSERIAL PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES brain.projects(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    content_type VARCHAR(50) NOT NULL, -- 'requirement', 'architecture', 'function_summary', etc.
    source_table VARCHAR(50),           -- Numele tabelei sursă (ex: 'architecture_decisions')
    source_id BIGINT,                  -- ID-ul din tabela sursă
    embedding VECTOR(768),             -- Dimensiune potrivită pentru nomic-embed-text (Ollama)
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 8. Index-uri pentru performanță
CREATE INDEX idx_memories_project_id ON brain.memories(project_id);

CREATE INDEX idx_memories_embedding ON brain.memories 
USING hnsw (embedding vector_cosine_ops);