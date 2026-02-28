import pandas as pd
df = pd.read_csv("/media/rvcse22/CSERV/medlm/medcasereasoning_core.csv")

# Check if case_prompt already contains diagnostic reasoning
sample = df.iloc[0]
print("=== CASE PROMPT ===")
print(sample["case_prompt"][:500])
print("\n=== DIAGNOSTIC REASONING ===")
print(sample["diagnostic_reasoning"][:300])
print("\n=== FINAL DIAGNOSIS ===")
print(sample["final_diagnosis"])

# Check for overlap
if sample["diagnostic_reasoning"][:50] in sample["case_prompt"]:
    print("\n⚠️ WARNING: Label leakage detected!")
else:
    print("\n✅ No leakage — case_prompt and diagnostic_reasoning are separate")
