#!/usr/bin/env python3

"""
India Runs - Data & AI Challenge
Offline Intelligent Candidate Ranking System

Run:
python rank.py

Input:
candidates.jsonl.gz

Output:
submission.csv
"""

import json
import csv
import argparse
import gzip
import os
from datetime import datetime, date


REFERENCE_DATE = date(2026, 6, 19)

CORE_REQUIRED_SKILLS = {
    "python","pytorch","tensorflow","scikit-learn",
    "llm","rag","nlp","transformers",
    "embeddings","vector database",
    "semantic search","retrieval",
    "ranking","faiss"
}

BONUS_SKILLS = {
    "docker","kubernetes","mlops",
    "langchain","sql","spark"
}

GOOD_TITLES = {
    "ai engineer",
    "ml engineer",
    "machine learning engineer",
    "data scientist",
    "software engineer"
}

PREFERRED_LOCATIONS = {
    "pune","delhi","noida",
    "bangalore","hyderabad","mumbai"
}


def load_candidates(path):
    file = gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, "r", encoding="utf-8")

    with file as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)



def score_skills(candidate):
    score = 0
    skills = candidate.get("skills", [])

    for skill in skills:
        name = str(skill.get("name","")).lower()

        if any(x in name for x in CORE_REQUIRED_SKILLS):
            score += 1

        elif any(x in name for x in BONUS_SKILLS):
            score += 0.3

    return min(score/10,1)



def score_career(candidate):
    title = str(candidate.get("profile",{}).get("current_title","")).lower()

    return 1 if any(x in title for x in GOOD_TITLES) else 0.3



def score_experience(candidate):
    years = candidate.get("profile",{}).get("years_of_experience",0)

    try:
        years=float(years)
    except:
        years=0

    if 6 <= years <= 8:
        return 1
    if 4 <= years < 6:
        return 0.8

    return 0.4



def score_behavior(candidate):

    signals=candidate.get("redrob_signals",{})

    score=0

    if signals.get("open_to_work_flag"):
        score += 0.3

    score += float(signals.get("recruiter_response_rate",0))*0.2

    return min(score,1)



def score_location(candidate):

    loc=str(candidate.get("profile",{}).get("location","")).lower()

    return 1 if any(x in loc for x in PREFERRED_LOCATIONS) else 0.5



def final_score(candidate):

    s1=score_skills(candidate)
    s2=score_career(candidate)
    s3=score_experience(candidate)
    s4=score_behavior(candidate)
    s5=score_location(candidate)

    score=(
        s1*0.35+
        s2*0.25+
        s3*0.15+
        s4*0.15+
        s5*0.10
    )

    profile=candidate.get("profile",{})

    reason=f"{profile.get('current_title','N/A')} | {profile.get('years_of_experience',0)} yrs experience"

    return round(score,4), reason



def main():

    parser=argparse.ArgumentParser()

    parser.add_argument("--candidates", default="candidates.jsonl.gz")
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--top", type=int, default=100)

    args=parser.parse_args()

    if not os.path.exists(args.candidates):
        print("Dataset missing:",args.candidates)
        return


    results=[]

    print("Processing candidates...")


    for candidate in load_candidates(args.candidates):

        score,reason=final_score(candidate)

        results.append(
            (
                candidate.get("candidate_id","UNKNOWN"),
                score,
                reason
            )
        )


    results.sort(key=lambda x:(-x[1],x[0]))

    with open(args.out,"w",newline="",encoding="utf-8") as f:

        writer=csv.writer(f)

        writer.writerow(
            ["candidate_id","rank","score","reasoning"]
        )

        for rank,row in enumerate(results[:args.top],1):

            writer.writerow(
                [row[0],rank,row[1],row[2]]
            )


    print("Done:",args.out)



if __name__=="__main__":
    main()
