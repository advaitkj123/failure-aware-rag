import json
from pathlib import Path
from generation.generate import generate_pair

OUTPUT_PATH = Path("data/processed/counterfactual_outputs.json")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---- SAFE, CURATED QUERIES ----
DATA = [
    {
        "qid": "q1",
        "query": "Who discovered penicillin?",
        "passages": [
            "Alexander Fleming discovered penicillin in 1928.",
            "Penicillin is an antibiotic used to treat bacterial infections."
        ],
    },
    {
        "qid": "q2",
        "query": "What is the capital of Australia?",
        "passages": [
            "Sydney is the largest city in Australia.",
            "Canberra is the capital city of Australia."
        ],
    },
    {
        "qid": "q3",
        "query": "Who wrote Hamlet?",
        "passages": [
            "William Shakespeare wrote the play Hamlet.",
            "Hamlet is a tragedy written sometime between 1599 and 1601."
        ],
    },
    {
        "qid": "q4",
        "query": "What year did the Titanic sink?",
        "passages": [
            "The RMS Titanic sank in the North Atlantic Ocean in 1912.",
            "The Titanic was a British passenger liner."
        ],
    },
    {
        "qid": "q5",
        "query": "Is Pluto a planet?",
        "passages": [
            "Pluto was reclassified as a dwarf planet by the IAU in 2006.",
            "Pluto was once considered the ninth planet."
        ],
    },
    {
        "qid": "q6",
        "query": "Who invented the telephone?",
        "passages": [
            "Alexander Graham Bell is credited with inventing the telephone.",
            "There were earlier devices for transmitting sound."
        ],
    },
    {
        "qid": "q7",
        "query": "What is the boiling point of water?",
        "passages": [
            "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
            "Boiling points can change with pressure."
        ],
    },
    {
        "qid": "q8",
        "query": "Who painted the Mona Lisa?",
        "passages": [
            "Leonardo da Vinci painted the Mona Lisa.",
            "The Mona Lisa is displayed in the Louvre Museum."
        ],
    },
    {
        "qid": "q9",
        "query": "What is the largest planet in the Solar System?",
        "passages": [
            "Jupiter is the largest planet in the Solar System.",
            "Saturn is known for its prominent ring system."
        ],
    },
    {
        "qid": "q10",
        "query": "Who was the first President of the United States?",
        "passages": [
            "George Washington was the first President of the United States.",
            "The United States gained independence in 1776."
        ],
    },
]

def main():
    results = []

    for item in DATA:
        print(f"Generating for {item['qid']}...")
        ans_no, ans_rag = generate_pair(item["query"], item["passages"])

        results.append({
            "qid": item["qid"],
            "query": item["query"],
            "passages": item["passages"],
            "answer_no_rag": ans_no,
            "answer_rag": ans_rag,
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[OK] Saved {len(results)} counterfactual examples to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
