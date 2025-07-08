def generate_suggestions(stage_counts):
    tips = []
    if stage_counts[2] > 0:
        tips.append("🚨 Urgent: Stage 3 tumor detected. Immediate consultation recommended.")
    if stage_counts[1] > 0:
        tips.append("⚠️ Monitor: Stage 2 tumor(s) present. Schedule a follow-up.")
    if stage_counts[0] > 0:
        tips.append("✅ Mild: Stage 1 tumor(s) detected. Keep monitoring regularly.")
    if sum(stage_counts) == 0:
        tips.append("🎉 No tumor detected. Stay healthy!")
    return tips
