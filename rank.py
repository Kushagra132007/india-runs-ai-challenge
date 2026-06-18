#!/usr/bin/env python3
"""
India Runs - Data & AI Challenge
Intelligent Candidate Ranking System

Job: Senior AI Engineer — Founding Team at Redrob AI
Approach: Multi-signal weighted scoring (no API calls, runs fully offline)

Scoring Components:
1. Core AI/ML Skills Match     (35%)
2. Career History & Title      (25%)
3. Experience Years            (15%)
4. Behavioral / Platform Signals (15%)
5. Education & Location        (10%)

Run:
    python rank.py --candidates candidates.jsonl --out submission.csv
"""

import json
import csv
import argparse
import math
from datetime import datetime, date

# ─────────────────────────────────────────────
# JOB REQUIREMENTS (derived from job_description.docx)
# ─────────────────────────────────────────────

# Must-have skills (high weight)
CORE_REQUIRED_SKILLS = {
    # Embeddings / Retrieval
    "sentence-transformers", "sentence transformers", "embeddings", "vector embeddings",
    "dense retrieval", "semantic search", "bge", "e5", "openai embeddings",
    # Vector DBs
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "faiss", "vector database", "vector search", "hybrid search",
    # Ranking / IR
    "ranking", "retrieval", "information retrieval", "reranking", "re-ranking",
    "learning to rank", "bm25", "ndcg", "mrr", "map",
    # LLMs & ML
    "llm", "large language models", "fine-tuning", "fine-tuning llms", "lora", "qlora",
    "peft", "rag", "retrieval augmented generation",
    "nlp", "natural language processing", "transformers",
    # Python & infra
    "python", "pytorch", "tensorflow", "scikit-learn", "xgboost",
    # Evaluation
    "a/b testing", "evaluation framework", "offline evaluation",
}

# Nice-to-have skills (lower weight)
BONUS_SKILLS = {
    "kafka", "spark", "airflow", "docker", "kubernetes",
    "distributed systems", "mlops", "mlflow", "weights & biases",
    "recommendation systems", "search", "ir", "hugging face",
    "langchain", "openai", "gpt", "bert", "t5",
    "data engineering", "sql", "nosql", "redis",
}

# Title signals — strong positive indicators
GOOD_TITLES = {
    "ai engineer", "ml engineer", "machine learning engineer",
    "senior ai engineer", "senior ml engineer",
    "nlp engineer", "data scientist", "applied scientist",
    "research engineer", "search engineer", "ranking engineer",
    "ai researcher", "ml researcher",
    "software engineer", "backend engineer", "full stack engineer",
    "senior software engineer", "principal engineer",
    "data engineer",  # some data engineers have strong ML background
}

# Title signals — negative / irrelevant (the "keyword stuffer" trap)
BAD_TITLES = {
    "marketing manager", "hr manager", "content writer", "graphic designer",
    "accountant", "sales executive", "customer support",
    "operations manager", "business analyst", "project manager",
    "civil engineer", "mechanical engineer", "product manager",
}

# Companies that are consulting-only (explicit disqualifier in JD)
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "ltimindtree", "mindtree",  # also listed
}

# Preferred locations
PREFERRED_LOCATIONS = {"pune", "noida", "delhi", "delhi ncr", "ncr", "hyderabad",
                        "mumbai", "bangalore", "bengaluru", "gurgaon", "gurugram"}

# ─────────────────────────────────────────────
# SCORING FUNCTIONS
# ─────────────────────────────────────────────

def score_skills(candidate: dict) -> float:
    """Score based on AI/ML skill match. Returns 0.0 - 1.0"""
    skills = candidate.get("skills", [])
    if not skills:
        return 0.0

    core_hits = 0
    bonus_hits = 0
    total_weighted = 0.0

    for skill in skills:
        name = skill.get("name", "").lower().strip()
        proficiency = skill.get("proficiency", "beginner")
        endorsements = skill.get("endorsements", 0)
        duration = skill.get("duration_months", 0)

        # Proficiency multiplier
        prof_mult = {"beginner": 0.4, "intermediate": 0.7, "advanced": 0.9, "expert": 1.0}.get(proficiency, 0.5)

        # Endorsement signal (log scale, capped)
        endorse_bonus = min(math.log1p(endorsements) / math.log1p(100), 0.3)

        # Duration bonus (capped at 48 months = 4 years)
        duration_bonus = min(duration / 48, 1.0) * 0.2

        skill_score = prof_mult + endorse_bonus + duration_bonus

        if any(req in name for req in CORE_REQUIRED_SKILLS) or name in CORE_REQUIRED_SKILLS:
            core_hits += 1
            total_weighted += skill_score * 2.0  # double weight for core skills
        elif any(bon in name for bon in BONUS_SKILLS) or name in BONUS_SKILLS:
            bonus_hits += 1
            total_weighted += skill_score * 0.5

    # Also check assessment scores from redrob_signals
    signals = candidate.get("redrob_signals", {})
    assessments = signals.get("skill_assessment_scores", {})
    for skill_name, score in assessments.items():
        if any(req in skill_name.lower() for req in CORE_REQUIRED_SKILLS):
            total_weighted += (score / 100) * 1.5

    # Normalize: ideal is 8+ core hits with good quality
    normalized = min(total_weighted / 20.0, 1.0)

    # Penalty if NO core skills at all
    if core_hits == 0:
        normalized *= 0.2

    return normalized


def score_career(candidate: dict) -> float:
    """Score based on career history and titles. Returns 0.0 - 1.0"""
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    current_title = profile.get("current_title", "").lower()

    total_score = 0.0
    all_consulting = True
    has_product_company = False
    ai_relevant_months = 0

    # Current title check
    if any(t in current_title for t in GOOD_TITLES):
        total_score += 0.35
    elif any(t in current_title for t in BAD_TITLES):
        total_score -= 0.2  # penalty for irrelevant title

    for job in career:
        title = job.get("title", "").lower()
        company = job.get("company", "").lower()
        description = job.get("description", "").lower()
        duration = job.get("duration_months", 0)
        company_size = job.get("company_size", "1-10")
        is_current = job.get("is_current", False)

        # Check if it's a consulting firm
        is_consulting = any(cf in company for cf in CONSULTING_FIRMS)
        if not is_consulting:
            all_consulting = False

        # Product company size signal
        size_map = {"1-10": 0, "11-50": 1, "51-200": 2, "201-500": 3,
                    "501-1000": 4, "1001-5000": 5, "5001-10000": 6, "10001+": 7}
        size_score = size_map.get(company_size, 0)
        # Prefer mid-sized product companies (not too small, not megacorp services)
        if 2 <= size_score <= 6:
            has_product_company = True

        # Title relevance
        title_score = 0.0
        if any(t in title for t in GOOD_TITLES):
            title_score = 1.0
        elif any(t in title for t in BAD_TITLES):
            title_score = -0.1

        # Description relevance — check for AI/ML keywords in actual work done
        desc_keywords = [
            "embedding", "vector", "ranking", "retrieval", "llm", "model",
            "nlp", "search", "recommendation", "neural", "transformer",
            "fine-tun", "pipeline", "inference", "production ml",
            "a/b test", "evaluation", "deployed", "real users"
        ]
        desc_hits = sum(1 for kw in desc_keywords if kw in description)
        desc_score = min(desc_hits / 5.0, 1.0)

        # Recent experience gets more weight
        recency_weight = 1.3 if is_current else 1.0

        job_score = (title_score * 0.5 + desc_score * 0.5) * recency_weight

        # Weight by duration (up to 36 months)
        duration_weight = min(duration / 36.0, 1.0)
        total_score += job_score * duration_weight * 0.15

        if title_score > 0 and desc_hits >= 2:
            ai_relevant_months += duration

    # Bonuses & penalties
    if all_consulting and len(career) > 1:
        total_score *= 0.5  # explicit JD disqualifier

    if has_product_company:
        total_score += 0.1

    if ai_relevant_months >= 36:
        total_score += 0.15  # 3+ years in AI roles

    return max(0.0, min(total_score, 1.0))


def score_experience(candidate: dict) -> float:
    """Score based on years of experience. Returns 0.0 - 1.0"""
    profile = candidate.get("profile", {})
    years = profile.get("years_of_experience", 0)

    # Target: 5-9 years (JD says 6-8 is ideal)
    if 5 <= years <= 9:
        # Peak at 6-8
        if 6 <= years <= 8:
            return 1.0
        return 0.85
    elif 4 <= years < 5:
        return 0.7  # JD says may consider 4+ if other signals strong
    elif 9 < years <= 12:
        return 0.75  # a bit over, still ok
    elif years > 12:
        return 0.55  # too senior, may not want to code
    elif 3 <= years < 4:
        return 0.5
    else:
        return 0.2  # too junior


def score_behavioral_signals(candidate: dict) -> float:
    """Score based on Redrob platform signals. Returns 0.0 - 1.0"""
    signals = candidate.get("redrob_signals", {})

    total = 0.0

    # Open to work — critical!
    if signals.get("open_to_work_flag", False):
        total += 0.25

    # Recency — last active (penalize inactive candidates heavily per JD)
    last_active_str = signals.get("last_active_date", "")
    if last_active_str:
        try:
            last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
            days_inactive = (date.today() - last_active).days
            if days_inactive <= 30:
                total += 0.25
            elif days_inactive <= 90:
                total += 0.15
            elif days_inactive <= 180:
                total += 0.05
            else:
                total -= 0.1  # 6+ months inactive = probably not available
        except Exception:
            pass

    # Recruiter response rate (JD explicitly mentions this)
    response_rate = signals.get("recruiter_response_rate", 0)
    total += response_rate * 0.2  # 0-0.20

    # Profile completeness
    completeness = signals.get("profile_completeness_score", 0) / 100
    total += completeness * 0.1

    # Interview completion rate
    interview_rate = signals.get("interview_completion_rate", 0)
    total += interview_rate * 0.1

    # Notice period — JD wants sub-30 days
    notice = signals.get("notice_period_days", 90)
    if notice <= 30:
        total += 0.1
    elif notice <= 60:
        total += 0.05

    # GitHub activity (positive signal for AI engineers)
    github = signals.get("github_activity_score", -1)
    if github >= 0:
        total += (github / 100) * 0.1

    # Saved by recruiters = market validation
    saved = signals.get("saved_by_recruiters_30d", 0)
    total += min(saved / 20, 0.05)

    # Verified contact info
    if signals.get("verified_email", False):
        total += 0.02
    if signals.get("verified_phone", False):
        total += 0.02
    if signals.get("linkedin_connected", False):
        total += 0.02

    return max(0.0, min(total, 1.0))


def score_education_location(candidate: dict) -> float:
    """Score education tier and location match. Returns 0.0 - 1.0"""
    education = candidate.get("education", [])
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    edu_score = 0.0
    for edu in education:
        tier = edu.get("tier", "unknown")
        field = edu.get("field_of_study", "").lower()
        degree = edu.get("degree", "").lower()

        # Tier score
        tier_map = {"tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.6,
                    "tier_4": 0.4, "unknown": 0.5}
        t_score = tier_map.get(tier, 0.5)

        # Relevant field bonus
        relevant_fields = ["computer science", "engineering", "information technology",
                           "mathematics", "statistics", "data science", "ai", "ml"]
        field_bonus = 0.2 if any(f in field for f in relevant_fields) else 0.0

        edu_score = max(edu_score, t_score + field_bonus)

    edu_score = min(edu_score, 1.0) * 0.6  # Education is 60% of this component

    # Location score
    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    willing_relocate = signals.get("willing_to_relocate", False)

    loc_score = 0.0
    if any(loc in location for loc in PREFERRED_LOCATIONS):
        loc_score = 1.0
    elif country == "india":
        loc_score = 0.6  # India but not preferred city — can relocate
    elif willing_relocate and country == "india":
        loc_score = 0.7
    elif willing_relocate:
        loc_score = 0.3  # foreign but willing to relocate (JD says case-by-case)
    else:
        loc_score = 0.1

    return edu_score + loc_score * 0.4


def compute_final_score(candidate: dict) -> tuple:
    """
    Compute weighted final score for a candidate.
    Returns (score, reasoning_string)
    """
    s_skills   = score_skills(candidate)
    s_career   = score_career(candidate)
    s_exp      = score_experience(candidate)
    s_behavior = score_behavioral_signals(candidate)
    s_edu_loc  = score_education_location(candidate)

    # Weighted combination (must sum to 1.0)
    final = (
        s_skills   * 0.35 +
        s_career   * 0.25 +
        s_exp      * 0.15 +
        s_behavior * 0.15 +
        s_edu_loc  * 0.10
    )

    # Build reasoning string
    profile   = candidate.get("profile", {})
    title     = profile.get("current_title", "N/A")
    yoe       = profile.get("years_of_experience", 0)
    signals   = candidate.get("redrob_signals", {})
    resp_rate = signals.get("recruiter_response_rate", 0)

    reasoning = (
        f"{title} with {yoe:.1f} yrs; "
        f"skills={s_skills:.2f}, career={s_career:.2f}, "
        f"exp={s_exp:.2f}, behavior={s_behavior:.2f}, "
        f"response rate={resp_rate:.2f}."
    )

    return round(final, 4), reasoning


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rank candidates for Senior AI Engineer role")
    parser.add_argument("--candidates", default="candidates.jsonl", help="Path to candidates.jsonl")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    parser.add_argument("--top", type=int, default=100, help="Number of candidates to output")
    args = parser.parse_args()

    print(f"Loading candidates from {args.candidates}...")
    candidates = []
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    print(f"Loaded {len(candidates):,} candidates. Scoring...")

    scored = []
    for i, cand in enumerate(candidates):
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1:,} / {len(candidates):,}...")
        score, reasoning = compute_final_score(cand)
        scored.append((cand["candidate_id"], score, reasoning))

    # Sort by score descending, ties broken by candidate_id ascending (per validation rules)
    scored.sort(key=lambda x: (-x[1], x[0]))

    top_candidates = scored[:args.top]

    print(f"\nTop 5 candidates:")
    for rank, (cid, score, reasoning) in enumerate(top_candidates[:5], 1):
        print(f"  #{rank}: {cid} | Score: {score:.4f} | {reasoning}")

    print(f"\nWriting output to {args.out}...")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (cid, score, reasoning) in enumerate(top_candidates, 1):
            writer.writerow([cid, rank, score, reasoning])

    print(f"Done! Wrote {len(top_candidates)} candidates to {args.out}")


if __name__ == "__main__":
    main()
