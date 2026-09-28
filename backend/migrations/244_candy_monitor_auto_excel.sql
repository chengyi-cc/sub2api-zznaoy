ALTER TABLE candy_monitor_settings ADD COLUMN IF NOT EXISTS auto_excel_on_incorrect BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE candy_monitor_accounts ADD COLUMN IF NOT EXISTS auto_excel_on_incorrect BOOLEAN;
