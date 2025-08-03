-- Seed data: Default persona configurations
-- Description: Insert default persona profiles for YMemo
-- Author: system
-- Created: 2025-01-02

-- Insert default personas (will be executed after persona table creation)
INSERT INTO public.ymemo_persona (name, description) VALUES
(
    'Michael Wei',
    'xxxxxx'
)
ON CONFLICT DO NOTHING;
