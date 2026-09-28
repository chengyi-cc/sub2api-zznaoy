package repository

// Merge activation defaults only when transitioning from native to Excel.
// The explicit request patch wins over defaults; unrelated stored fields survive.
func excelBPSActivationSQL(merged, patch string) string {
	return "(CASE WHEN platform='openai' AND type='oauth' AND parent_account_id IS NULL" +
		" AND extra->'openai_excel_bps' IS DISTINCT FROM 'true'::jsonb" +
		" AND (" + patch + ")->'openai_excel_bps'='true'::jsonb THEN (" + merged + ")" +
		" || ('{\"base_rpm\":15,\"openai_rpm_overflow\":true,\"openai_excel_bps_cache_creation_as_input\":true,\"openai_excel_bps_auto_disable_on_403\":true}'::jsonb || " + patch + ")" +
		" || jsonb_build_object('openai_excel_bps_last_transition',jsonb_build_object('reason','manual','at',NOW()))" +
		" WHEN extra->'openai_excel_bps'='true'::jsonb AND (" + patch + ")->'openai_excel_bps'='false'::jsonb" +
		" THEN (" + merged + ") || jsonb_build_object('openai_excel_bps_last_transition',jsonb_build_object('reason','manual_disabled','at',NOW()))" +
		" ELSE (" + merged + ") END)"
}
