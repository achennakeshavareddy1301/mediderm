
An AI system that analyzes skin images and brief symptom text to generate clinically grounded medication reports with clear instructions, precautions, and follow-up guidance.

What the system does

Intake: Captures a skin photo and short patient narrative (symptoms, duration, prior medication), in local language or text.

Analysis: Uses a vision encoder (e.g., ViT) to extract lesion features and an LLM to interpret symptom text; an alignment module projects image embeddings into the LLM token space for joint reasoning.

Diagnosis hinting: Produces a shortlist of likely skin conditions with confidence and key visual findings; supports multi-condition reasoning when features overlap.

Medication report: Generates evidence-based recommendations for topical/systemic options, dosing, application frequency, contraindications, and red flags, formatted for patients and clinicians.

Explainability: Adds “why this recommendation” sections derived from visual cues and symptom matches to help clinical review and patient understanding.

Safety rails: Detects dangerous combinations (e.g., steroid overuse), flags uncertain cases for referral, and includes follow-up checkpoints.


<img width="6778" height="3066" alt="p drawio" src="https://github.com/user-attachments/assets/eac110b0-6335-4751-a104-7e2a8397ca66" />

