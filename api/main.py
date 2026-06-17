"""
api/main.py
FastAPI application for the Physics Solver (5-stage redesign).

Endpoints:
  POST /solve         — solve a physics problem
  POST /solve/batch   — solve multiple problems (up to 10)
  GET  /health        — health check
  GET  /graph/stats   — graph statistics

Run: uvicorn api.main:app --reload --port 8000
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from collections import Counter

from solver.graph_loader import load_graphs
from solver.pipeline     import PhysicsSolver


solver: PhysicsSolver | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global solver
    print("Loading graph index...")
    graph  = load_graphs()
    solver = PhysicsSolver(graph)
    print("Physics Solver (5-stage) ready.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="JEE/NEET Physics Solver",
    description=(
        "LLM-driven conceptual equation selection + SymPy exact arithmetic. "
        "5-stage pipeline: parse → frontier resolution → SymPy → narrate → distractors."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class SolveRequest(BaseModel):
    question: str = Field(
        ..., min_length=10,
        example=(
            "A body of density 8000 kg/m³ and volume 0.5 m³ moves from "
            "rest and reaches 30 m/s after traveling 40 m. Find the net force."
        ),
    )

class BatchSolveRequest(BaseModel):
    questions: list[str] = Field(..., max_length=10)


@app.get("/health")
def health():
    return {"status": "ok", "solver_ready": solver is not None, "version": "2.0"}


@app.get("/graph/stats")
def graph_stats():
    if solver is None:
        raise HTTPException(503, "Solver not initialized")
    g = solver.graph
    domain_counts = dict(Counter(n["domain"] for n in g.nodes))
    return {
        "total_equations": len(g.nodes),
        "total_edges":     len(g.edges),
        "domains":         domain_counts,
    }


@app.post("/solve")
def solve(req: SolveRequest):
    """
    Solve a JEE/NEET physics numerical question.

    Returns:
    - answer.value / answer.value_exact / answer.unit
    - explanation  — teacher-style step-by-step narration
    - decision_log — what the LLM chose each round and why
    - chain_summary — equation chain in execution order
    - mcq          — 4-option MCQ with correct + 3 distractors
    - confidence   — HIGH (SymPy-verified) or UNVERIFIED
    """
    if solver is None:
        raise HTTPException(503, "Solver not initialized")

    response = solver.solve(req.question)

    return {
        "question":       response.question,
        "answer": {
            "value":       response.final_value,
            "value_exact": response.final_value_exact,
            "unit":        response.final_unit,
            "symbol":      response.final_symbol,
        },
        "explanation":    response.explanation,
        "decision_log":   response.decision_log,
        "chain_summary":  response.chain_summary,
        "mcq":            response.mcq,
        "confidence":     response.confidence,
        "time_s":         response.time_taken_s,
        "error":          response.error or None,
    }


@app.post("/solve/batch")
def solve_batch(req: BatchSolveRequest):
    if solver is None:
        raise HTTPException(503, "Solver not initialized")
    results = []
    for q in req.questions:
        r = solver.solve(q)
        results.append({
            "question":      q,
            "answer":        {"value": r.final_value, "unit": r.final_unit,
                              "value_exact": r.final_value_exact},
            "confidence":    r.confidence,
            "chain_summary": r.chain_summary,
            "error":         r.error or None,
        })
    return {"results": results}
