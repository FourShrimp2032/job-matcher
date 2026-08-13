from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field



class CandidateSkill(BaseModel):
    name: str
    level: Literal["beginner", "intermediate", "advanced", "unknown"] = "unknown"
    evidence: str = ""
    capabilities: list[str] = Field(default_factory=list)


class CandidateAIProfile(BaseModel):
    skills: list[CandidateSkill] = Field(default_factory=list)
    experience_years: float | None = None
    english_level: str | None = None
    education: list[str] = Field(default_factory=list)
    summary: str


class JobRequirement(BaseModel):
    skill: str
    importance: Literal["required", "preferred", "optional"] = "required"
    description: str = ""
    capabilities: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)


class JobAIProfile(BaseModel):
    title: str | None = None
    seniority: str | None = None
    experience_years_required: float | None = None
    english_level: str | None = None
    requirements: list[JobRequirement] = Field(default_factory=list)
    summary: str


class MatchAIExplanation(BaseModel):
    summary: str
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    interview_focus: list[str] = Field(default_factory=list)



class CandidateCreate(BaseModel):
    name: str
    email: str | None = None
    cv_text: str = Field(min_length=30)


class CandidateRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    email: str | None
    cv_text: str
    profile: dict
    created_at: datetime


class JobCreate(BaseModel):
    company: str | None = None
    title: str | None = None
    description: str = Field(min_length=30)
    url: str | None = None


class JobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    company: str | None
    title: str
    description: str
    url: str | None
    status: str
    profile: dict
    created_at: datetime


class MatchCreate(BaseModel):
    candidate_id: int
    job_id: int


class MatchRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    candidate_id: int
    job_id: int
    score: float
    recommendation: str
    details: dict
    created_at: datetime
