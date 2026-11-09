"""Lapisan API demo MedIR. Logika retrieval ada di engine.py."""
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from engine import Engine


class TestQuery(BaseModel):
    id: str
    teks: str
    n_relevan: int


class SearchRequest(BaseModel):
    query: str = Field(default="", max_length=500)
    pakai_cross_encoder: bool = True
    split_test_id: Optional[str] = None


class Sinyal(BaseModel):
    kode: str
    nama: str
    bobot: float
    skor_ternormalisasi: float
    kontribusi: float


class Hasil(BaseModel):
    peringkat: int
    doc_id: str
    judul: str
    cuplikan: str
    skor: float
    memuat_kata_query: bool
    kata_cocok: list[str]
    kata_tampil: list[str]
    punya_label_train: bool
    relevansi: Optional[int]
    sinyal: Optional[list[Sinyal]]


class Metrik(BaseModel):
    p5: float
    p10: float
    ndcg10: float


class MetrikPerbandingan(BaseModel):
    bm25: Metrik
    final: Metrik


class SearchResponse(BaseModel):
    query: str
    split_test_id: Optional[str]
    pakai_cross_encoder: bool
    kata_query: list[str]
    n_relevan: Optional[int]
    waktu_ms: float
    bm25: list[Hasil]
    final: list[Hasil]
    metrik: Optional[MetrikPerbandingan]


class UjiFinalVsExpansion(BaseModel):
    selisih_p5: float
    p_wilcoxon: float
    naik: int
    turun: int
    sama: int


class SummaryResponse(BaseModel):
    final_table: dict[str, Any]
    error_analysis: dict[str, Any]
    final_vs_expansion: UjiFinalVsExpansion


class Health(BaseModel):
    status: str


state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["engine"] = Engine()
    state["summary"] = state["engine"].summary()
    yield
    state.clear()


app = FastAPI(title="MedIR Demo", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health", response_model=Health)
def health():
    return {"status": "siap"}


@app.get("/api/queries/test", response_model=list[TestQuery])
def test_queries():
    return state["engine"].test_queries()


@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if req.split_test_id is None and not req.query.strip():
        raise HTTPException(status_code=422, detail="Query tidak boleh kosong.")
    try:
        return state["engine"].search(req.query.strip(), req.split_test_id, req.pakai_cross_encoder)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Query test '{req.split_test_id}' tidak ditemukan.")


@app.get("/api/summary", response_model=SummaryResponse)
def summary():
    return state["summary"]
