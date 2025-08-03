-- Migration: 003_add_updated_at_to_ymemo.sql
-- Description: Add updated_at field to ymemo table for consistency with persona table
-- Author: system
-- Created: 2025-08-02
-- Dependencies: 001_baseline_ymemo_table.sql
--
-- This migration adds the updated_at timestamp field to the existing ymemo table,
-- following the same pattern as the ymemo_persona table. It includes:
-- 1. Adding the updated_at column with default value (safe for existing records)
-- 2. Backfilling existing records with created_at timestamp
-- 3. Creating trigger function for automatic updates
-- 4. Creating trigger for automatic timestamp updates on UPDATE operations
--
-- ROLLBACK: To rollback this migration:
-- DROP TRIGGER IF EXISTS trigger_ymemo_updated_at ON public.ymemo;
-- DROP FUNCTION IF EXISTS update_ymemo_updated_at();
-- ALTER TABLE public.ymemo DROP COLUMN IF EXISTS updated_at;

-- Step 1: Add updated_at column with default value
-- This is safe for existing data - all current records will get NOW() as default
-- Handle potential ownership issues by using IF NOT EXISTS pattern
DO $$
BEGIN
    -- Check if column already exists
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='ymemo' AND column_name='updated_at') THEN
        ALTER TABLE public.ymemo
        ADD COLUMN updated_at timestamp with time zone NOT NULL DEFAULT NOW();
    END IF;
END $$;

-- Step 2: Backfill existing records to set updated_at = created_at
-- This gives existing records a logical updated_at value
DO $$
BEGIN
    -- Only update if the column was just added (updated_at equals default NOW())
    UPDATE public.ymemo
    SET updated_at = created_at
    WHERE ABS(EXTRACT(EPOCH FROM (updated_at - NOW()))) < 5;
END $$;

-- Step 3: Create trigger function for automatic updated_at updates
-- This function will be called on every UPDATE to set updated_at = NOW()
CREATE OR REPLACE FUNCTION update_ymemo_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Step 4: Create trigger to automatically update updated_at on record updates
-- This ensures updated_at is always current when a record is modified
DO $$
BEGIN
    -- Drop trigger if it exists, then create it
    DROP TRIGGER IF EXISTS trigger_ymemo_updated_at ON public.ymemo;
    CREATE TRIGGER trigger_ymemo_updated_at
        BEFORE UPDATE ON public.ymemo
        FOR EACH ROW
        EXECUTE FUNCTION update_ymemo_updated_at();
END $$;

-- Step 5: Add index for performance (optional but recommended)
-- This helps with queries that filter or sort by updated_at
DO $$
BEGIN
    -- Create index if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname = 'idx_ymemo_updated_at') THEN
        CREATE INDEX idx_ymemo_updated_at ON public.ymemo (updated_at DESC);
    END IF;
END $$;
